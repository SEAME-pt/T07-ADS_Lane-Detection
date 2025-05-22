#include "LaneDetector.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <opencv2/imgproc.hpp>

/**
 * @brief Constructor for LaneDetector class.
 * @param trt_model_path Path to the TensorRT model file.
 * @throws std::runtime_error If CUDA stream creation or model loading fails.
 */
LaneDetector::LaneDetector(const std::string& trt_model_path) {
    // Create a CUDA stream for asynchronous GPU operations
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA stream creation failed: " + std::string(cudaGetErrorString(err)));
    }

    // Initialize Kalman filter with 4 state variables (offset, offset velocity, angle, angle velocity)
    // and 2 measurements (offset, angle)
    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    // Measurement matrix: Maps state to measurements (offset and angle are directly measured)
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);
    // Transition matrix: Models state evolution over time (includes velocity for smoothing)
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 1, 0, 0,
                                                        0, 1, 0, 0,
                                                        0, 0, 1, 1,
                                                        0, 0, 0, 1);
    // Process noise covariance: Small value for stable tracking
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(0.03));
    // Measurement noise covariance: Moderate value to trust measurements
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1.0));
    // Initial error covariance: Start with some uncertainty
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0));
    // Initialize matrices for Kalman filter measurements and predictions
    measurement_ = cv::Mat(2, 1, CV_32F);
    prediction_ = cv::Mat(4, 1, CV_32F);

    // Initialize Kalman-filtered offset and angle to zero
    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    // Set initial lane edges (assuming a lane width of ~200px centered at 320px)
    last_left_edge_ = 220.0f;  // Initial left edge (100px left of center)
    last_right_edge_ = 420.0f; // Initial right edge (100px right of center)

    // Load the TensorRT model for lane detection
    loadEngine(trt_model_path);
}

/**
 * @brief Destructor for LaneDetector class.
 * Cleans up CUDA resources to prevent memory leaks.
 */
LaneDetector::~LaneDetector() {
    // Destroy CUDA stream if it exists
    if (stream_) cudaStreamDestroy(stream_);
    // Free CUDA memory for input buffer
    if (buffers_[0]) cudaFree(buffers_[0]);
    // Free CUDA memory for output buffer
    if (buffers_[1]) cudaFree(buffers_[1]);
}

/**
 * @brief Initializes the camera capture pipeline using GStreamer.
 * @return True if the pipeline opens successfully, false otherwise.
 */
bool LaneDetector::initialize() {
    // Define GStreamer pipeline to capture video from the camera
    // - Resolution: 640x360, format: NV12, framerate: 30fps
    // - Converts to BGR format for OpenCV processing
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                          "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                          "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    // Open the camera using the GStreamer pipeline
    cap_.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        std::cerr << "Error opening GStreamer pipeline" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Loads the TensorRT model and sets up inference resources.
 * @param trt_model_path Path to the TensorRT model file.
 * @throws std::runtime_error If model loading or CUDA memory allocation fails.
 */
void LaneDetector::loadEngine(const std::string& trt_model_path) {
    // Open the TensorRT model file in binary mode
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Error opening TensorRT model file: " + trt_model_path);
    }

    // Read the model file into a vector
    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Create TensorRT runtime, engine, and execution context for inference
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr));
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create TensorRT execution context");

    // Allocate CUDA memory for input (1x3x160x320) and output (1x1x160x320) buffers
    cudaError_t err = cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(buffers_[0]);
        throw std::runtime_error("CUDA malloc failed for output buffer: " + std::string(cudaGetErrorString(err)));
    }

    // Resize host-side input and output data vectors to match buffer sizes
    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

/**
 * @brief Preprocesses the input frame for TensorRT inference.
 * @param frame Input frame (expected 640x360, CV_8UC3).
 * @throws std::runtime_error If the frame is invalid.
 */
void LaneDetector::preprocess(const cv::Mat& frame) {
    // Validate input frame: Must be non-empty, BGR (CV_8UC3), and 640x360
    if (frame.empty() || frame.type() != CV_8UC3 || frame.cols != frame_width_ || frame.rows != frame_height_) {
        throw std::runtime_error("Invalid input frame: expected " + std::to_string(frame_width_) + "x" + 
                                 std::to_string(frame_height_) + " CV_8UC3");
    }

    // Crop the frame to the region of interest (ROI: y=40 to y=360, 640x320)
    // This focuses on the lower part of the frame where the lane is likely to be
    cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_);
    cv::Mat cropped = frame(roi);

    // Convert to grayscale to estimate brightness and apply CLAHE
    cv::Mat gray;
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);

    // Estimate brightness by computing the mean intensity (0-255)
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];

    // Apply CLAHE to enhance contrast in low-light conditions
    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(brightness < 100 ? 4.0 : 2.0); // Higher clip limit for darker images
    clahe->setTilesGridSize(cv::Size(8, 8)); // 8x8 tiles for adaptive equalization
    if (brightness < 150) { // Apply CLAHE if the image is relatively dark
        cv::Mat lab;
        cv::cvtColor(cropped, lab, cv::COLOR_BGR2Lab); // Convert to Lab color space
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels); // Split into L, a, b channels
        clahe->apply(lab_channels[0], lab_channels[0]); // Apply CLAHE to L channel (lightness)
        cv::merge(lab_channels, lab); // Merge channels back
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR); // Convert back to BGR
    } else {
        enhanced = cropped; // Use the original cropped frame if brightness is sufficient
    }

    // Resize to match the model input size (320x160)
    cv::Mat resized;
    cv::resize(enhanced, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

    // Normalize pixel values to [0, 1] for model input
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // Split into BGR channels and copy to input data buffer in CHW format (required by TensorRT)
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    for (int c = 0; c < 3; ++c) {
        float* dst = input_data_.data() + c * input_height_ * input_width_;
        memcpy(dst, channels[c].ptr<float>(), input_width_ * input_height_ * sizeof(float));
    }
}

/**
 * @brief Runs TensorRT inference on the preprocessed frame.
 * @throws std::runtime_error If CUDA operations or inference fails.
 */
void LaneDetector::infer() {
    // Copy input data from host to device (GPU) memory asynchronously
    cudaError_t err = cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float), 
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to device failed: " + std::string(cudaGetErrorString(err)));

    // Execute TensorRT inference on the GPU
    if (!context_->enqueueV2(buffers_, stream_, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    // Copy inference output from device to host memory asynchronously
    err = cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float), 
                          cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to host failed: " + std::string(cudaGetErrorString(err)));

    // Synchronize the CUDA stream to ensure all operations are complete
    cudaStreamSynchronize(stream_);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA error after inference: " + std::string(cudaGetErrorString(err)));
}

/**
 * @brief Calculates lane geometry (offset and angle) for steering.
 * @param offset Output: Distance from the frame center to the lane center (px).
 * @param angle Output: Angle of the lane centerline (degrees).
 * @throws std::runtime_error If mask size is invalid.
 */
void LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
    const int camera_center = frame_width_ / 2; // Frame center: 320 (for 640px width)
    const int max_distance = 300; // Maximum distance to search for lane edges from the center
    std::vector<cv::Point2f> centers; // Store lane center points for fitting a line
    std::vector<float> weights; // Weights for weighted least squares (higher weights for lower rows)

    // Convert lane mask to 8-bit for edge detection
    cv::Mat mask_8u;
    lane_mask_.convertTo(mask_8u, CV_8U, 255.0);
    // Validate mask size matches frame dimensions
    if (mask_8u.cols != frame_width_ || mask_8u.rows != frame_height_) {
        throw std::runtime_error("Mask size mismatch: expected " + std::to_string(frame_width_) + "x" + 
                                 std::to_string(frame_height_) + ", got " + 
                                 std::to_string(mask_8u.cols) + "x" + std::to_string(mask_8u.rows));
    }

    // Detect lane edges at sampled rows (y=350 to y=40, step 10)
    for (int y = roi_end_y_ - 10; y >= roi_start_y_; y -= 10) {
        uchar* row = mask_8u.ptr<uchar>(y);
        int left = camera_center, right = camera_center;

        // Search left from the center for the left lane edge
        for (int x = camera_center; x >= std::max(0, camera_center - max_distance); --x) {
            if (row[x] > 128) { // Threshold to detect lane pixel
                left = x;
                break;
            }
        }
        // Search right from the center for the right lane edge
        for (int x = camera_center; x < std::min(frame_width_, camera_center + max_distance); ++x) {
            if (row[x] > 128) {
                right = x;
                break;
            }
        }

        // If both edges are detected and the lane width is reasonable (>50px), compute the lane center
        if (left != camera_center && right != camera_center && right - left > 50) {
            float cx = (left + right) / 2.0f; // Center x-coordinate
            centers.emplace_back(cx, y); // Store center point
            // Weight the point based on its y-position (lower rows are more important)
            float w = static_cast<float>(y - roi_start_y_) / (roi_end_y_ - roi_start_y_);
            weights.push_back(w * w);
            // Update last known edges at y=350 for interpolation in processFrame
            if (y == roi_end_y_ - 10) {
                last_left_edge_ = left;
                last_right_edge_ = right;
            }
        }
    }

    // If fewer than 3 center points are detected, use the last Kalman-filtered values
    if (centers.size() < 3) {
        offset = offset_kalman_;
        angle = angle_kalman_;
        return;
    }

    // Fit a line to the center points using weighted least squares to compute offset and angle
    float sum_w = 0, sum_y = 0, sum_x = 0, sum_yx = 0, sum_yy = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        float w = weights[i];
        float x = centers[i].x;
        float y = centers[i].y;
        sum_w += w;
        sum_x += w * x;
        sum_y += w * y;
        sum_yx += w * y * x;
        sum_yy += w * y * y;
    }

    // Compute line parameters (y = ax + b) using weighted least squares
    float denom = sum_w * sum_yy - sum_y * sum_y;
    float a = 0.0f, b = 0.0f;
    if (std::abs(denom) > 1e-5f) {
        a = (sum_w * sum_yx - sum_y * sum_x) / denom; // Slope
        b = (sum_x * sum_yy - sum_y * sum_yx) / denom; // Intercept
    }

    // Compute offset at the reference y (y=350)
    float y_ref = roi_end_y_ - 10;
    float x_ref = a * y_ref + b;
    offset = x_ref - camera_center; // Offset: Distance from frame center (positive = right, negative = left)
    angle = std::atan(a) * 180.0f / CV_PI; // Angle: Slope converted to degrees (positive = right turn)

    // Update Kalman filter with the new measurements
    measurement_.at<float>(0) = offset;
    measurement_.at<float>(1) = angle;
    kalman_.correct(measurement_);
    prediction_ = kalman_.predict();
    offset_kalman_ = prediction_.at<float>(0); // Smoothed offset
    angle_kalman_ = prediction_.at<float>(2); // Smoothed angle

    // Clamp offset and angle to reasonable ranges to prevent extreme steering commands
    offset = std::clamp(offset_kalman_, -frame_width_ / 2.0f, frame_width_ / 2.0f);
    angle = std::clamp(angle_kalman_, -45.0f, 45.0f);
}


/**
 * @brief Processes a frame to detect lanes and compute geometry.
 * @param frame Input frame (expected 640x360, CV_8UC3).
 * @param offset Output: Distance from the frame center to the lane center (px).
 * @param angle Output: Angle of the lane centerline (degrees).
 * @param output_frame Output: Frame with visualizations (green mask, yellow points, magenta centerline, etc.).
 * @param visualize_mask If true, displays a thumbnail of the lane mask in the top-right corner.
 */
void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    // Preprocess the frame (crop, enhance, resize, normalize) for TensorRT inference
    preprocess(frame);
    // Run inference to get the lane mask (probability map of lane pixels)
    infer();

    // Convert model output to a probability map using sigmoid: P(lane) = 1 / (1 + exp(-x))
    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

    // Estimate brightness to adjust the lane detection threshold
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0]; // 0-255
    // Use a lower threshold in darker conditions to detect more lane pixels
    float threshold = brightness < 100 ? 0.3 : 0.5;
    cv::threshold(lane_mask_, lane_mask_, threshold, 1.0, cv::THRESH_BINARY);

    // Apply morphological closing to fill small gaps in the lane mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(lane_mask_, lane_mask_, cv::MORPH_CLOSE, kernel);

    // Resize the lane mask back to the original frame size (640x360)
    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_NEAREST);

    // Clone the input frame to create the output frame with visualizations
    output_frame = frame.clone();

    // Calculate lane geometry (offset and angle) for steering
    calculateLaneGeometry(offset, angle);

    // Create a visualization of the lane mask with interpolated edges
    cv::Mat mask_vis;
    lane_mask_.convertTo(mask_vis, CV_8U, 255.0); // Convert mask to 8-bit for edge detection
    cv::Mat mask_color = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3); // BGR image for green overlay
    int camera_center = frame_width_ / 2; // 320
    int max_distance = 300; // Maximum distance to search for edges

    // Detect lane edges across multiple rows (y=350 to y=40, step 10)
    std::vector<int> left_edges, right_edges;
    std::vector<int> valid_y;
    for (int y = roi_end_y_ - 10; y >= roi_start_y_; y -= 10) {
        uchar* row = mask_vis.ptr<uchar>(y);
        int left = camera_center, right = camera_center;

        // Search left for the left lane edge with a broader window if needed
        for (int x = camera_center; x >= std::max(0, camera_center - max_distance); --x) {
            if (row[x] > 128) {
                left = x;
                break;
            }
        }
        // Search right for the right lane edge with a broader window if needed
        for (int x = camera_center; x < std::min(frame_width_, camera_center + max_distance); ++x) {
            if (row[x] > 128) {
                right = x;
                break;
            }
        }
        // Store edges if at least one is detected (-1 if not detected)
        if (left != camera_center || right != camera_center) {
            left_edges.push_back(left != camera_center ? left : -1);
            right_edges.push_back(right != camera_center ? right : -1);
            valid_y.push_back(y);
        }
    }

    // Initialize with last known edges and Kalman predictions if no new edges are detected
    int last_valid_left = last_left_edge_;
    int last_valid_right = last_right_edge_;
    float avg_lane_width = (last_valid_right - last_valid_left > 50) ? (last_valid_right - last_valid_left) : 200.0f; // Default to 200px if invalid
    if (!left_edges.empty() && !right_edges.empty()) {
        // Update last valid edges and average lane width
        for (size_t i = 0; i < left_edges.size(); ++i) {
            if (left_edges[i] != -1) last_valid_left = left_edges[i];
            if (right_edges[i] != -1) last_valid_right = right_edges[i];
            if (left_edges[i] != -1 && right_edges[i] != -1 && right_edges[i] - left_edges[i] > 50) {
                avg_lane_width = 0.9 * avg_lane_width + 0.1 * (right_edges[i] - left_edges[i]); // Exponential moving average
            }
        }
    }

    // Ensure edges are always defined using Kalman and trend-based extrapolation
    if (left_edges.empty() || right_edges.empty()) {
        // Use Kalman-predicted offset and angle to estimate edges
        float x_ref = camera_center + offset_kalman_;
        float lane_angle_rad = angle_kalman_ * CV_PI / 180.0f;
        for (int y = roi_end_y_ - 10; y >= roi_start_y_; y -= 10) {
            float dy = (y - (roi_end_y_ - 10));
            float dx = dy * tan(lane_angle_rad);
            int estimated_center = static_cast<int>(x_ref + dx);
            left_edges.push_back(std::max(0, estimated_center - static_cast<int>(avg_lane_width / 2)));
            right_edges.push_back(std::min(frame_width_ - 1, estimated_center + static_cast<int>(avg_lane_width / 2)));
            valid_y.push_back(y);
        }
    } else {
        // Interpolate or extrapolate missing edges using Kalman and trend
        for (size_t i = 0; i < left_edges.size(); ++i) {
            if (left_edges[i] == -1) {
                // Estimate left edge using Kalman and trend from previous valid
                if (i > 0 && left_edges[i-1] != -1) {
                    float slope = (last_valid_left - left_edges[i-1]) / (valid_y[i-1] - valid_y[i]);
                    left_edges[i] = static_cast<int>(left_edges[i-1] + slope * (valid_y[i-1] - valid_y[i]));
                } else {
                    float x_ref = camera_center + offset_kalman_;
                    float lane_angle_rad = angle_kalman_ * CV_PI / 180.0f;
                    float dy = (valid_y[i] - (roi_end_y_ - 10));
                    float dx = dy * tan(lane_angle_rad);
                    int estimated_center = static_cast<int>(x_ref + dx);
                    left_edges[i] = std::max(0, estimated_center - static_cast<int>(avg_lane_width / 2));
                }
            } else {
                last_valid_left = left_edges[i];
            }
            if (right_edges[i] == -1) {
                // Estimate right edge using Kalman and trend from previous valid
                if (i > 0 && right_edges[i-1] != -1) {
                    float slope = (last_valid_right - right_edges[i-1]) / (valid_y[i-1] - valid_y[i]);
                    right_edges[i] = static_cast<int>(right_edges[i-1] + slope * (valid_y[i-1] - valid_y[i]));
                } else {
                    float x_ref = camera_center + offset_kalman_;
                    float lane_angle_rad = angle_kalman_ * CV_PI / 180.0f;
                    float dy = (valid_y[i] - (roi_end_y_ - 10));
                    float dx = dy * tan(lane_angle_rad);
                    int estimated_center = static_cast<int>(x_ref + dx);
                    right_edges[i] = std::min(frame_width_ - 1, estimated_center + static_cast<int>(avg_lane_width / 2));
                }
            } else {
                last_valid_right = right_edges[i];
            }
            // Ensure minimum lane width
            if (right_edges[i] - left_edges[i] < 50) {
                right_edges[i] = left_edges[i] + 50;
            }
        }
    }

    // Draw green horizontal lines between detected or estimated edges
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y = valid_y[i];
        int left = left_edges[i];
        int right = right_edges[i];
        if (right - left < 50) right = left + 50; // Minimum width
        for (int x = left; x <= right && x < frame_width_; ++x) {
            mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green
        }
        last_left_edge_ = left;
        last_right_edge_ = right;
    }

    // Fill gaps between sampled rows for a smoother green overlay
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y_start = valid_y[i];
        int y_end = (i + 1 < valid_y.size()) ? valid_y[i + 1] : y_start;
        int left_start = left_edges[i];
        int right_start = right_edges[i];
        int left_end = (i + 1 < valid_y.size()) ? left_edges[i + 1] : left_start;
        int right_end = (i + 1 < valid_y.size()) ? right_edges[i + 1] : right_start;

        for (int y = y_start; y <= y_end; ++y) {
            float t = (y_end == y_start) ? 0.0f : static_cast<float>(y - y_start) / (y_end - y_start);
            int left = left_start + static_cast<int>(t * (left_end - left_start));
            int right = right_start + static_cast<int>(t * (right_end - right_start));
            if (right - left < 50) right = left + 50;
            for (int x = left; x <= right && x < frame_width_; ++x) {
                mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green
            }
        }
    }

    // Overlay the green mask on the output frame
    cv::addWeighted(output_frame, 0.7, mask_color, 0.5, 0.0, output_frame);

    // Draw yellow center points and magenta centerline
    std::vector<cv::Point2f> centers;
    std::vector<float> weights;
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y = valid_y[i];
        int left = left_edges[i];
        int right = right_edges[i];
        if (right - left > 50) {
            float cx = (left + right) / 2.0f;
            centers.emplace_back(cx, y);
            float w = static_cast<float>(y - roi_start_y_) / (roi_end_y_ - roi_start_y_);
            weights.push_back(w * w);
        }
    }
    for (const auto& pt : centers) {
        cv::circle(output_frame, pt, 4, cv::Scalar(0, 255, 255), -1);
    }
    if (centers.size() >= 3) {
        float sum_w = 0, sum_y = 0, sum_x = 0, sum_yx = 0, sum_yy = 0;
        for (size_t i = 0; i < centers.size(); ++i) {
            float w = weights[i];
            float x = centers[i].x;
            float y = centers[i].y;
            sum_w += w;
            sum_x += w * x;
            sum_y += w * y;
            sum_yx += w * y * x;
            sum_yy += w * y * y;
        }
        float denom = sum_w * sum_yy - sum_y * sum_y;
        if (std::abs(denom) > 1e-5f) {
            float a = (sum_w * sum_yx - sum_y * sum_x) / denom;
            float b = (sum_x * sum_yy - sum_y * sum_yx) / denom;
            int y_top = roi_end_y_ - 100;
            cv::Point pt1(a * y_top + b, y_top);
            cv::Point pt2(a * (roi_end_y_ - 10) + b, roi_end_y_ - 10);
            cv::line(output_frame, pt1, pt2, cv::Scalar(255, 0, 255), 2);
        }
    }

    // Draw blue line, yellow distance line, and text
    int lane_center = frame_width_ / 2 + static_cast<int>(offset);
    int line_y = (roi_start_y_ + roi_end_y_) / 2;
    cv::line(output_frame, cv::Point(lane_center, line_y), cv::Point(lane_center, line_y - 50), cv::Scalar(0, 0, 255), 3);
    int frame_center = frame_width_ / 2;
    int distance = std::abs(static_cast<int>(offset));
    cv::line(output_frame, cv::Point(frame_center, frame_height_ - 1), cv::Point(lane_center, frame_height_ - 2), cv::Scalar(0, 255, 255), 2);
    std::string distance_text = "Distance: " + std::to_string(distance) + " px";
    cv::putText(output_frame, distance_text, cv::Point(frame_center + offset / 2 - 50, frame_height_ - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw cyan lane width lines and text (bottom)
    int left_edge_bottom = left_edges[0];
    int right_edge_bottom = right_edges[0];
    if (right_edge_bottom - left_edge_bottom < 50) right_edge_bottom = left_edge_bottom + 50;
    cv::line(output_frame, cv::Point(left_edge_bottom, frame_height_ - 1), cv::Point(right_edge_bottom, frame_height_ - 1), cv::Scalar(255, 255, 0), 2);
    int lane_width_bottom = right_edge_bottom - left_edge_bottom;
    std::string width_text_bottom = "Lane Width (Bottom): " + std::to_string(lane_width_bottom) + " px";
    cv::putText(output_frame, width_text_bottom, cv::Point(left_edge_bottom + (right_edge_bottom - left_edge_bottom) / 2 - 50, frame_height_ - 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw cyan lane width line (center, y=200)
    int center_y = (roi_start_y_ + roi_end_y_) / 2; // Center of the screen, approximately y=200
    int left_edge_center = camera_center;
    int right_edge_center = camera_center;
    float t = 0.0f;
    size_t center_index = 0;

    // Find the nearest sampled row to the center (y=200)
    for (size_t i = 0; i < valid_y.size(); ++i) {
        if (valid_y[i] <= center_y) {
            center_index = i;
            break;
        }
    }
    if (center_index > 0 && center_index < valid_y.size()) {
        // Interpolate between the two nearest sampled rows
        float y1 = valid_y[center_index - 1];
        float y2 = valid_y[center_index];
        t = (center_y - y1) / (y2 - y1); // Interpolation factor
        left_edge_center = static_cast<int>(left_edges[center_index - 1] + t * (left_edges[center_index] - left_edges[center_index - 1]));
        right_edge_center = static_cast<int>(right_edges[center_index - 1] + t * (right_edges[center_index] - right_edges[center_index - 1]));
    } else if (center_index == 0 && !valid_y.empty()) {
        // Use the first sampled row if center is above or at the first point
        left_edge_center = left_edges[0];
        right_edge_center = right_edges[0];
    } else if (center_index == valid_y.size() && !valid_y.empty()) {
        // Use the last sampled row if center is below the last point
        left_edge_center = left_edges.back();
        right_edge_center = right_edges.back();
    }

    // Draw the cyan line and text if edges are detected and width is reasonable
    if (right_edge_center - left_edge_center < 50) right_edge_center = left_edge_center + 50;
    cv::line(output_frame, cv::Point(left_edge_center, center_y), cv::Point(right_edge_center, center_y), cv::Scalar(255, 255, 0), 2);
    int lane_width_center = right_edge_center - left_edge_center;
    std::string width_text_center = "Lane Width (Center): " + std::to_string(lane_width_center) + " px";
    cv::putText(output_frame, width_text_center, cv::Point(left_edge_center + (right_edge_center - left_edge_center) / 2 - 50, center_y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw offset and angle text
    std::string offset_text = "Offset: " + std::to_string(static_cast<int>(offset)) + " px";
    cv::putText(output_frame, offset_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::string angle_text = "Angle: " + std::to_string(static_cast<int>(angle)) + " deg";
    cv::putText(output_frame, angle_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    // Output mask thumbnail if requested
    if (visualize_mask) {
        cv::Mat mask_thumb;
        cv::resize(mask_vis, mask_thumb, cv::Size(frame_width_ / 4, frame_height_ / 4), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(mask_thumb, mask_thumb, cv::COLOR_GRAY2BGR);
        mask_thumb.copyTo(output_frame(cv::Rect(frame_width_ - mask_thumb.cols, 0, mask_thumb.cols, mask_thumb.rows)));
    }
}