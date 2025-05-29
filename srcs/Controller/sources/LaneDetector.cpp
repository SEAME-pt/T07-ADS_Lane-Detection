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

    // input_height_ = 128;
    // input_width_ = 256;
    // frame_height_ = 128; // Corrected to match input frame
    // frame_width_ = ;  // Corrected to match input frame
    roi_start_y_ = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252
    roi_end_y_ = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360

    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = 200.0f;
    prev_left_edge_ = frame_width_ / 2;  // 320
    prev_right_edge_ = frame_width_ / 2; // 320
    last_left_edge_ = frame_width_ / 2;  // 320
    last_right_edge_ = frame_width_ / 2; // 320

    loadEngine(trt_model_path);
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
    return cap_.isOpened();
}

bool LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
    // Check if lane mask is valid
    if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
        return false;
    }

    // Step 1: Define the Region of Interest (ROI)
    int start_y, end_y, start_x, end_x;
    defineROI(start_y, end_y, start_x, end_x);
    cv::Rect roi(start_x, start_y, end_x - start_x, end_y - start_y);

    // Step 2: Find left and right lane edges using dense sampling
    std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi, left_edges, right_edges)) {
        return false; // Not enough edge points detected
    }

    // Step 3: Perform weighted linear regression to fit lines to edges
    double left_slope, left_intercept, right_slope, right_intercept;
    weightedLinearRegression(left_edges, left_slope, left_intercept);
    weightedLinearRegression(right_edges, right_slope, right_intercept);
    std::cout << "Left  Line: slope = " << left_slope << " | intercept = " << left_intercept << " || " << "Right Line: slope = " << right_slope << " | intercept = " << right_intercept << std::endl;

    // Step 4: Calculate offset and angle from the fitted lines
    float measured_offset, measured_angle;
    calculateOffsetAndAngle(left_slope, left_intercept, right_slope, right_intercept,
                            end_y - 1, measured_offset, measured_angle);

    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;

    return true;
}

void LaneDetector::defineROI(int& start_y, int& end_y, int& start_x, int& end_x) const {
    start_y = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252 for 360
    end_y = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
    start_x = 0;
    end_x = frame_width_; // 640
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

void LaneDetector::drawDebugInfo(cv::Mat* debug_img,
                                 const std::vector<cv::Point>& left_edges,
                                 const std::vector<cv::Point>& right_edges,
                                 float offset, float angle) const {
    if (debug_img->empty()) {
        *debug_img = cv::Mat(frame_height_, frame_width_, CV_8UC3, cv::Scalar(0)); // 640x360
    }
    if (debug_img->type() != CV_8UC3) {
        debug_img->convertTo(*debug_img, CV_8UC3);
    }

    for (const auto& pt : left_edges) {
        cv::circle(*debug_img, pt, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto& pt : right_edges) {
        cv::circle(*debug_img, pt, 2, cv::Scalar(0, 255, 0), -1);
    }

    int y_bottom = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT) - 1; // 359
    int x_center = frame_width_ / 2; // 320
    int x_mid = x_center + static_cast<int>(offset / METER_PER_PIXEL);
    cv::line(*debug_img, cv::Point(x_mid, y_bottom),
             cv::Point(x_mid - static_cast<int>(100 * std::tan(angle)), y_bottom - 100),
             cv::Scalar(255, 255, 0), 2);
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

void LaneDetector::infer() {
    cudaError_t err = cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to device failed: " + std::string(cudaGetErrorString(err)));

    if (!context_->enqueueV2(buffers_, stream_, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    err = cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to host failed: " + std::string(cudaGetErrorString(err)));

    cudaStreamSynchronize(stream_);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA error after inference: " + std::string(cudaGetErrorString(err)));
}

void LaneDetector::preprocess(const cv::Mat& frame) {
    if (frame.empty() || frame.type() != CV_8UC3 || frame.cols != frame_width_ || frame.rows != frame_height_) {
        throw std::runtime_error("Invalid input frame: expected " + std::to_string(frame_width_) + "x" +
                                 std::to_string(frame_height_) + " CV_8UC3");
    }

    cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_); // 640x108
    cv::Mat cropped = frame(roi);

    cv::Mat gray;
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(brightness < 100 ? 4.0 : 2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));
    if (brightness < 150) {
        cv::Mat lab;
        cv::cvtColor(cropped, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    } else {
        enhanced = cropped;
    }

    cv::Mat resized;
    cv::resize(enhanced, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
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

    // test if true or false
    if(!calculateLaneGeometry(offset, angle))
    {
       std::cout << "failed to calculate lane geometry" << std::endl;
    }

    cv::Mat mask_vis;
    lane_mask_.convertTo(mask_vis, CV_8U, 255.0);
    cv::Mat mask_color = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
    int camera_center = frame_width_ / 2; // 320
    int max_distance = 300;

    std::vector<int> left_edges, right_edges;
    std::vector<int> valid_y;
    for (int y = roi_end_y_ - 10; y >= roi_start_y_; y -= 10) {
        uchar* row = mask_vis.ptr<uchar>(y);
        int left = camera_center, right = camera_center;

        for (int x = camera_center; x >= std::max(0, camera_center - max_distance); --x) {
            if (row[x] > 128) {
                left = x;
                break;
            }
        }
        for (int x = camera_center; x < std::min(frame_width_, camera_center + max_distance); ++x) {
            if (row[x] > 128) {
                right = x;
                break;
            }
        }
        if (left != camera_center || right != camera_center) {
            left_edges.push_back(left != camera_center ? left : -1);
            right_edges.push_back(right != camera_center ? right : -1);
            valid_y.push_back(y);
        }
    }

    int last_valid_left = last_left_edge_;
    int last_valid_right = last_right_edge_;
    float avg_lane_width = (last_valid_right - last_valid_left > 50) ? (last_valid_right - last_valid_left) : 200.0f;
    if (!left_edges.empty() && !right_edges.empty()) {
        for (size_t i = 0; i < left_edges.size(); ++i) {
            if (left_edges[i] != -1) last_valid_left = left_edges[i];
            if (right_edges[i] != -1) last_valid_right = right_edges[i];
            if (left_edges[i] != -1 && right_edges[i] != -1 && right_edges[i] - left_edges[i] > 50) {
                avg_lane_width = 0.9 * avg_lane_width + 0.1 * (right_edges[i] - left_edges[i]);
            }
        }
    }

    if (left_edges.empty() || right_edges.empty()) {
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
        for (size_t i = 0; i < left_edges.size(); ++i) {
            if (left_edges[i] == -1) {
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
            if (right_edges[i] - left_edges[i] < 50) {
                right_edges[i] = left_edges[i] + 50;
            }
        }
    }

    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y = valid_y[i];
        int left = left_edges[i];
        int right = right_edges[i];
        if (right - left < 50) right = left + 50;
        for (int x = left; x <= right && x < frame_width_; ++x) {
            mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
        last_left_edge_ = left;
        last_right_edge_ = right;
    }

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
                mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
            }
        }
    }

    cv::addWeighted(output_frame, 0.7, mask_color, 0.5, 0.0, output_frame);

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

    int lane_center = frame_width_ / 2 + static_cast<int>(offset);
    int line_y = (roi_start_y_ + roi_end_y_) / 2;
    cv::line(output_frame, cv::Point(lane_center, line_y), cv::Point(lane_center, line_y - 50), cv::Scalar(0, 0, 255), 3);
    int frame_center = frame_width_ / 2;
    int distance = std::abs(static_cast<int>(offset));
    cv::line(output_frame, cv::Point(frame_center, frame_height_ - 1), cv::Point(lane_center, frame_height_ - 2), cv::Scalar(0, 255, 255), 2);
    std::string distance_text = "Distance: " + std::to_string(distance) + " px";
    cv::putText(output_frame, distance_text, cv::Point(frame_center + offset / 2 - 50, frame_height_ - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    int left_edge_bottom = left_edges[0];
    int right_edge_bottom = right_edges[0];
    if (right_edge_bottom - left_edge_bottom < 50) right_edge_bottom = left_edge_bottom + 50;
    cv::line(output_frame, cv::Point(left_edge_bottom, frame_height_ - 1), cv::Point(right_edge_bottom, frame_height_ - 1), cv::Scalar(255, 255, 0), 2);
    int lane_width_bottom = right_edge_bottom - left_edge_bottom;
    std::string width_text_bottom = "Lane Width (Bottom): " + std::to_string(lane_width_bottom) + " px";
    cv::putText(output_frame, width_text_bottom, cv::Point(left_edge_bottom + (right_edge_bottom - left_edge_bottom) / 2 - 50, frame_height_ - 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    int center_y = (roi_start_y_ + roi_end_y_) / 2;
    int left_edge_center = camera_center;
    int right_edge_center = camera_center;
    float t = 0.0f;
    size_t center_index = 0;

    for (size_t i = 0; i < valid_y.size(); ++i) {
        if (valid_y[i] <= center_y) {
            center_index = i;
            break;
        }
    }
    if (center_index > 0 && center_index < valid_y.size()) {
        float y1 = valid_y[center_index - 1];
        float y2 = valid_y[center_index];
        t = (center_y - y1) / (y2 - y1);
        left_edge_center = static_cast<int>(left_edges[center_index - 1] + t * (left_edges[center_index] - left_edges[center_index - 1]));
        right_edge_center = static_cast<int>(right_edges[center_index - 1] + t * (right_edges[center_index] - right_edges[center_index - 1]));
    } else if (center_index == 0 && !valid_y.empty()) {
        left_edge_center = left_edges[0];
        right_edge_center = right_edges[0];
    } else if (center_index == valid_y.size() && !valid_y.empty()) {
        left_edge_center = left_edges.back();
        right_edge_center = right_edges.back();
    }

    std::cout << "angle: " << angle << " offset: " << offset << std::endl;

    if (right_edge_center - left_edge_center < 50) right_edge_center = left_edge_center + 50;
    cv::line(output_frame, cv::Point(left_edge_center, center_y), cv::Point(right_edge_center, center_y), cv::Scalar(255, 255, 0), 2);
    int lane_width_center = right_edge_center - left_edge_center;
    std::string width_text_center = "Lane Width (Center): " + std::to_string(lane_width_center) + " px";
    cv::putText(output_frame, width_text_center, cv::Point(left_edge_center + (right_edge_center - left_edge_center) / 2 - 50, center_y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    std::string offset_text = "Offset: " + std::to_string(static_cast<int>(offset)) + " px";
    cv::putText(output_frame, offset_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::string angle_text = "Angle: " + std::to_string(static_cast<int>(angle)) + " deg";
    cv::putText(output_frame, angle_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    if (visualize_mask) {
        cv::Mat mask_thumb;
        cv::resize(mask_vis, mask_thumb, cv::Size(frame_width_ / 4, frame_height_ / 4), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(mask_thumb, mask_thumb, cv::COLOR_GRAY2BGR);
        mask_thumb.copyTo(output_frame(cv::Rect(frame_width_ - mask_thumb.cols, 0, mask_thumb.cols, mask_thumb.rows)));
    }
}