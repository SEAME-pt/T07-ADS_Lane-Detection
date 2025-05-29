#include "LaneDetector.hpp"

LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);
    
    // Initialize Kalman filter
    kf_ = cv::KalmanFilter(2, 2, 0, CV_32F);
    kf_.statePre.at<float>(0) = 0.0f;
    kf_.statePre.at<float>(1) = 0.0f;
    kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPre, cv::Scalar::all(1));
    
    input_height_ = 128;
    input_width_ = 256;
    frame_height_ = 360; // Corrected to match input frame
    frame_width_ = 640;  // Corrected to match input frame
    roi_sy_ = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252
    roi_ey_ = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
    
    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = 200.0f;

    prev_left_edge_ = frame_width_ / 2;  // 320
    prev_right_edge_ = frame_width_ / 2; // 320
    last_left_edge_ = frame_width_ / 2;  // 320
    last_right_edge_ = frame_width_ / 2; // 320
    
	defineROI();
    
    debug_ = std::make_unique<Debug>(frame_width_, frame_height_, roi_sy_, roi_ey_);
    loadEngine(trt_model_path);
	std::cout << "LaneDetector created with model: " << trt_model_path << std::endl;
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool LaneDetector::initialize() {
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);
	if (!cap_.isOpened()) {
		std::cerr << "Failed to open camera pipeline!" << std::endl;
		return false;
	}
	std::cout << "[" << __func__ << "] "
				<< "Camera pipeline opened successfully: \n"
				<< pipeline << std::endl;
	std::cout << "[" << __func__ << "] "
				<< "LaneDetector initialization concluded!"
				<< std::endl;
    return cap_.isOpened();
}

bool LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
    // Check if lane mask is valid
    if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
        return false;
    }

    // Step 1: Define the Region of Interest (ROI)
    cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);

    // Step 2: Find left and right lane edges using dense sampling
    std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi, left_edges, right_edges)) {
		std::cerr << "Not enough edge points detected in ROI!" << std::endl;
        return false; // Not enough edge points detected
    }

    left_edges_ = left_edges;
    right_edges_ = right_edges;
    // Step 3: Perform weighted linear regression to fit lines to edges
    double left_slope, left_intercept, right_slope, right_intercept;
    weightedLinearRegression(left_edges, left_slope, left_intercept);
    weightedLinearRegression(right_edges, right_slope, right_intercept);
    //std::cout << "Left  Line: slope = " << left_slope << " | intercept = " << left_intercept << " || " << "Right Line: slope = " << right_slope << " | intercept = " << right_intercept << std::endl;

    // Step 4: Calculate offset and angle from the fitted lines
    float measured_offset, measured_angle;
    calculateOffsetAndAngle(left_slope, left_intercept, right_slope, right_intercept,
                            roi_ey_ - 1, measured_offset, measured_angle);
	std::cout << "[" << __func__ << "] "
			  << "Measured Offset: " << std::setw(6) << measured_offset
			  << " m, Measured Angle: " << std::setw(6) << measured_angle << " rad"
			  << '\r' << std::flush;


	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;
    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;
	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;

    return true;
}

void LaneDetector::defineROI() {
		std::cout << "Defining ROI..." << std::endl;
		roi_sy_ = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252 for 360
		roi_ey_ = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
		roi_sx_ = ROI_X_BORDER;
		roi_ex_ = frame_width_ - ROI_X_BORDER; // 640
		std::cout << "ROI: "
					<< "sy = " << roi_sy_
					<< ", ey = " << roi_ey_
					<< ", sx = " << roi_sx_
					<< ", ex = " << roi_ex_
					<< std::endl;
}

bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi,
                                 std::vector<cv::Point>& left_edges,
                                 std::vector<cv::Point>& right_edges) const {
    left_edges.clear();
    right_edges.clear();

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        int left_x = -1, right_x = -1;
        for (int x = roi.x; x < roi.x + roi.width && x < roi.x + MAX_SEARCH_DISTANCE; ++x) {
            if (lane_mask.at<float>(y, x) > 0.5f) {
                left_x = x;
                break;
            }
        }
        for (int x = roi.x + roi.width - 1; x >= roi.x && x >= roi.x + roi.width - MAX_SEARCH_DISTANCE; --x) {
            if (lane_mask.at<float>(y, x) > 0.5f) {
                right_x = x;
                break;
            }
        }
        if (left_x != -1 && right_x != -1 && left_x < right_x) {
            left_edges.emplace_back(left_x, y);
            right_edges.emplace_back(right_x, y);
        }
    }
    return !left_edges.empty() && !right_edges.empty();
}

void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& points,
                                            double& slope, double& intercept) const {
    if (points.size() < 2) {
        slope = 0.0;
        intercept = frame_width_ / 2.0; // 320
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int start_y = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252
    int end_y = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
    double range_y = end_y - start_y;

    for (const auto& pt : points) {
        double y = pt.y;
        double x = pt.x;
        double weight = (y - start_y) / range_y;
        weight = std::max(0.1, weight);

        sum_w += weight;
        sum_wy += weight * y;
        sum_wx += weight * x;
        sum_wyy += weight * y * y;
        sum_wyx += weight * y * x;
    }

    double mean_y = sum_wy / sum_w;
    double mean_x = sum_wx / sum_w;
    double denom = sum_wyy - 2 * mean_y * sum_wy + sum_w * mean_y * mean_y;
    if (std::abs(denom) < 1e-6) {
        slope = 0.0;
        intercept = mean_x;
    } else {
        slope = (sum_wyx - mean_y * sum_wx - mean_x * sum_wy + sum_w * mean_x * mean_y) / denom;
        intercept = mean_x - slope * mean_y;
    }
}

void LaneDetector::calculateOffsetAndAngle(double left_slope, double left_intercept,
                                           double right_slope, double right_intercept,
                                           int y_bottom, float& offset, float& angle) const {
    double x_left = left_slope * y_bottom + left_intercept;
    double x_right = right_slope * y_bottom + right_intercept;
    double x_mid = (x_left + x_right) / 2.0;
    double x_center = frame_width_ / 2.0; // 320

    float offset_pixels = static_cast<float>(x_mid - x_center);
    offset = offset_pixels * METER_PER_PIXEL;

    double avg_slope = (left_slope + right_slope) / 2.0;
    float angle_image = std::atan(avg_slope);
    // angle = angle_image - static_cast<float>(CAMERA_TILT);
    angle = angle_image;
    //std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;

}

void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle,
                                     float& smoothed_offset, float& smoothed_angle) {
    cv::Mat prediction = kf_.predict();
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured_offset, measured_angle);
    cv::Mat corrected = kf_.correct(measurement);
    smoothed_offset = corrected.at<float>(0);
    smoothed_angle = corrected.at<float>(1);
}

void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Error opening TensorRT model file: " + trt_model_path);
    }

    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr));
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create TensorRT execution context");

    cudaError_t err = cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float)); // 1x3x128x256
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float)); // 1x1x128x256
    if (err != cudaSuccess) {
        cudaFree(buffers_[0]);
        throw std::runtime_error("CUDA malloc failed for output buffer: " + std::string(cudaGetErrorString(err)));
    }

    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

/**
 * @brief Runs inference using the TensorRT engine.
 *
 * This function performs the complete inference pipeline:
 * - Copies input data from host (CPU) to device (GPU).
 * - Executes inference asynchronously using TensorRT.
 * - Copies the output data back from device to host.
 * - Synchronizes the CUDA stream to ensure all operations are complete.
 * - Checks for any CUDA errors during the process.
 *
 * @throws std::runtime_error if any CUDA memory transfer fails,
 *         if inference execution fails, or if there are post-inference CUDA errors.
 */
void LaneDetector::infer() {
    // Copy input data from host (CPU) to device (GPU) memory asynchronously.
    // buffers_[0] is the input buffer on the GPU.
    cudaError_t err = cudaMemcpyAsync(
        buffers_[0],						// Destination: GPU input buffer
        input_data_.data(),					// Source: CPU input data
        input_data_.size() * sizeof(float),	// Number of bytes to copy
        cudaMemcpyHostToDevice,				// Direction: Host to Device
        stream_								// CUDA stream to execute this copy
    );
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to device failed: " + std::string(cudaGetErrorString(err)));

    // Run inference on the GPU using TensorRT.
    // 'enqueueV2' schedules the execution of the network on the GPU using provided buffers and stream.
    if (!context_->enqueueV2(buffers_, stream_, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    // Copy the inference output data back from device (GPU) to host (CPU) memory asynchronously.
    // buffers_[1] is the output buffer on the GPU.
    err = cudaMemcpyAsync(
        output_data_.data(),				// Destination: CPU output buffer
        buffers_[1],						// Source: GPU output buffer
        output_data_.size() * sizeof(float),// Number of bytes to copy
        cudaMemcpyDeviceToHost,				// Direction: Device to Host
        stream_								// CUDA stream used for the copy
    );
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to host failed: " + std::string(cudaGetErrorString(err)));

    // Wait for all tasks in the CUDA stream (inference + memory transfers) to complete.
    cudaStreamSynchronize(stream_);

    // Check for any errors that may have occurred during kernel execution or memory operations.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error after inference: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * @brief Preprocesses an input frame for lane detection inference.
 *
 * This function performs several tasks:
 * - Validates the input frame format and size.
 * - Crops a region of interest (ROI).
 * - Enhances image contrast using CLAHE if necessary.
 * - Resizes the image to the model's input size.
 * - Normalizes pixel values to [0, 1] range.
 * - Reorders data into channel-first (CHW) format expected by the model.
 *
 * @param frame The input BGR image frame from a video stream or camera.
 *
 * @throws std::runtime_error if the input frame is invalid.
 */
void LaneDetector::preprocess(const cv::Mat& frame) {
    // === Input Validation ===
    // Check if the input frame is empty (not loaded or captured properly)
    if (frame.empty()) {
        std::cerr << "Error: Input frame is empty." << std::endl;
        throw std::runtime_error("Invalid input frame: frame is empty");
    }

    // Ensure the input frame is of expected 8-bit 3-channel (BGR) type
    if (frame.type() != CV_8UC3) {
        std::cerr << "Error: Input frame has wrong type. Expected CV_8UC3, got type " << frame.type() << "." << std::endl;
        throw std::runtime_error("Invalid input frame: incorrect type, expected CV_8UC3");
    }

    // Validate frame dimensions match what the model expects
    if (frame.cols != frame_width_ || frame.rows != frame_height_) {
        std::cerr << "Error: Input frame has incorrect dimensions. Expected "
                  << frame_width_ << "x" << frame_height_ << ", got "
                  << frame.cols << "x" << frame.rows << "." << std::endl;
        throw std::runtime_error("Invalid input frame: incorrect dimensions");
    }

    // === ROI (Region of Interest) Cropping ===
    // Define a rectangle using configured coordinates
    cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);

    // Crop the input image to the specified ROI
    cv::Mat cropped_roi = frame(roi);

    // === Brightness Estimation ===
    // Convert cropped image to grayscale to compute mean brightness
    cv::Mat gray;
    cv::cvtColor(cropped_roi, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];  // Use only the first channel (gray value)

    // === Contrast Enhancement (Adaptive) ===
    cv::Mat enhanced;

    // Create a CLAHE object for adaptive contrast enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    // Adjust contrast more aggressively if brightness is low
    clahe->setClipLimit(brightness < 100 ? 4.0 : 2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));  // Use an 8x8 tile grid

    if (brightness < 150) {
        // Convert cropped BGR image to Lab color space (L = lightness)
        cv::Mat lab;
        cv::cvtColor(cropped_roi, lab, cv::COLOR_BGR2Lab);

        // Split Lab image into separate channels
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);

        // Apply CLAHE to the L (lightness) channel to enhance local contrast
        clahe->apply(lab_channels[0], lab_channels[0]);

        // Merge channels back together and convert back to BGR
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    } else {
        // If brightness is sufficient, skip contrast enhancement
        enhanced = cropped_roi;
    }

    // === Resize to Model Input Size ===
    // Resize image to the dimensions expected by the model (e.g., 224x224)
    cv::Mat resized;
    cv::resize(enhanced, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

    // === Normalize Pixel Values ===
    // Convert pixel values from [0, 255] uchar to [0.0, 1.0] float
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // === Convert from HWC to CHW Format ===
    // Many models (e.g., TensorRT, PyTorch) expect channel-first layout
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);  // Split into B, G, R channels

    // Rearrange and copy each channel to the input buffer in CHW order
    for (int c = 0; c < 3; ++c) {
        float* dst = input_data_.data() + c * input_height_ * input_width_;
        memcpy(dst, channels[c].ptr<float>(), input_width_ * input_height_ * sizeof(float));
    }
}

void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];
    float threshold = brightness < 100 ? 0.3 : 0.5;
    cv::threshold(lane_mask_, lane_mask_, threshold, 1.0, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(lane_mask_, lane_mask_, cv::MORPH_CLOSE, kernel);

    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_NEAREST);

    output_frame = frame.clone();
    debug_->showOutputVideo(output_frame, left_edges_, right_edges_, offset, angle, lane_mask_, visualize_mask);

    // test if true or false
    if(!calculateLaneGeometry(offset, angle))
    {
       std::cout << "[" << __func__ <<"] "
	   				<< "Failed to calculate lane geometry" << std::endl;
    }

}