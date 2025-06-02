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
	// std::string pipeline = "nvarguscamerasrc exposuretimerange=\"1000000 50000000\" gainrange=\"1 16\" !"
	//                        "video/x-raw(memory:NVMM), width=640, height=360, "
    //                        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    //                        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
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
    // std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi, left_edges_, right_edges_)) {
		std::cerr << "Not enough edge points detected in ROI!" << std::endl;
        return false; // Not enough edge points detected
    }

    // Step 3: Perform weighted linear regression to fit lines to edges
    double left_slope, left_intercept, right_slope, right_intercept;
    weightedLinearRegression(left_edges_, left_slope, left_intercept);
    weightedLinearRegression(right_edges_, right_slope, right_intercept);
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

	std::cout << "[" << __func__ << "] "
			  << "Offset: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_offset
			  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << offset << ") m"
			  << "Angle: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_angle
			  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << smoothed_angle << ") rad"
			  << '\r' << std::flush;


    return true;
}

void LaneDetector::defineROI() {
		std::cout << "Defining ROI..." << std::endl;
		roi_sy_ = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252 for 360
		roi_ey_ = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
		roi_sx_ = ROI_X_BORDER; // 20
		roi_ex_ = frame_width_ - ROI_X_BORDER; // 620
		roi_w_ = roi_ex_ - roi_sx_; // 620 - 20 = 600
		roi_h_ = roi_ey_ - roi_sy_; // 360 - 252 = 108
		std::cout << "ROI: "
					<< "sy = " << roi_sy_
					<< ", ey = " << roi_ey_
					<< ", sx = " << roi_sx_
					<< ", ex = " << roi_ex_
					<< ", w = " << roi_w_
					<< ", h = " << roi_h_
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

void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges,
                                            double& slope, double& intercept) const {
    if (edges.size() < 2) {
        slope = 0.0;
        intercept = frame_width_ / 2.0; // 320
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int start_y = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252
    int end_y = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360
    double range_y = end_y - start_y;

    for (const auto& pt : edges) {
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


// yellow detected ??
void LaneDetector::preprocess(const cv::Mat& frame) {
    if (frame.empty() || frame.type() != CV_8UC3 || frame.cols != frame_width_ || frame.rows != frame_height_) {
        throw std::runtime_error("Invalid input frame: expected " + std::to_string(frame_width_) + "x" +
                                 std::to_string(frame_height_) + " CV_8UC3");
    }

    if (roi_sy_ < 0 || roi_ey_ <= roi_sy_ || roi_ey_ > frame_height_ || frame_width_ <= 0) {
        throw std::runtime_error("Invalid ROI: roi_sy_=" + std::to_string(roi_sy_) +
                                 ", roi_ey_=" + std::to_string(roi_ey_) +
                                 ", frame_width_=" + std::to_string(frame_width_) +
                                 ", frame_height_=" + std::to_string(frame_height_));
    }

    cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);
    // cv::Rect roi(0, roi_sy_, frame_width_, roi_ey_ - roi_sy_);
    cv::Mat cropped = frame(roi);
    // std::cout << "[preprocess] : 1 - Cropped frame size: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::Mat gray;
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];
    // std::cout << "[preprocess] : 2 - Mean brightness: " << brightness << std::endl;

    cv::Mat gamma_corrected = cropped;
    if (brightness < 100) {
        cv::Mat lookup(1, 256, CV_8U);
        float gamma = 0.6; // More aggressive brightening
        for (int i = 0; i < 256; ++i) {
            lookup.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        cv::LUT(cropped, lookup, gamma_corrected);
    }

    cv::Mat lab, enhanced_lab;
    cv::cvtColor(gamma_corrected, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab, lab_channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(brightness < 100 ? 6.0 : 2.5);
    clahe->setTilesGridSize(cv::Size(8, 8));
    clahe->apply(lab_channels[0], lab_channels[0]);
    clahe->apply(lab_channels[1], lab_channels[1]);
    clahe->apply(lab_channels[2], lab_channels[2]);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, enhanced_lab, cv::COLOR_Lab2BGR);
    // std::cout << "[preprocess] : 3 - Enhanced Lab image size: " << enhanced_lab.cols << "x" << enhanced_lab.rows << std::endl;

    cv::Mat hsv, yellow_mask;
    cv::cvtColor(gamma_corrected, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(10, 50, 50), cv::Scalar(50, 255, 255), yellow_mask); // Wider yellow range
    // std::cout << "[preprocess] : 4 - Yellow mask size: " << yellow_mask.cols << "x" << yellow_mask.rows << std::endl;

    std::vector<cv::Mat> hsv_channels(3);
    cv::split(hsv, hsv_channels);
    clahe->apply(hsv_channels[2], hsv_channels[2]);
    cv::merge(hsv_channels, hsv);
    cv::Mat enhanced_hsv;
    cv::cvtColor(hsv, enhanced_hsv, cv::COLOR_HSV2BGR);

    cv::Mat enhanced, yellow_enhanced;
    cv::bitwise_and(enhanced_lab, enhanced_lab, yellow_enhanced, yellow_mask);
    cv::addWeighted(enhanced_lab, 0.6, enhanced_hsv, 0.4, 0.0, enhanced); // More HSV weight for yellow
    cv::bitwise_and(enhanced, enhanced, enhanced, ~yellow_mask);
    cv::add(yellow_enhanced, enhanced, enhanced);
    // std::cout << "[preprocess] : 5 - Combined enhanced image size: " << enhanced.cols << "x" << enhanced.rows << std::endl;

    cv::Mat resized;
    cv::resize(enhanced, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);
    // std::cout << "[preprocess] : 6 - Resized to: " << input_width_ << "x" << input_height_ << std::endl;

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    // std::cout << "[preprocess] : 7 - Normalized size: " << normalized.cols << "x" << normalized.rows << ", type: " << normalized.type() << std::endl;

    size_t expected_size = 3 * input_width_ * input_height_;
    if (input_data_.size() < expected_size) {
        throw std::runtime_error("input_data_ size too small: " + std::to_string(input_data_.size()) +
                                 ", expected: " + std::to_string(expected_size));
    }

    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    for (int c = 0; c < 3; ++c) {
        if (!channels[c].isContinuous()) {
            throw std::runtime_error("Channel " + std::to_string(c) + " is not continuous");
        }
        float* dst = input_data_.data() + c * input_width_ * input_height_;
        memcpy(dst, channels[c].ptr<float>(), input_width_ * input_height_ * sizeof(float));
        // std::cout << "[preprocess] : 8 - Copied channel " << c << std::endl;
    }
}

//yellow not detected:
// void LaneDetector::preprocess(const cv::Mat& frame) {
//     if (frame.empty() || frame.type() != CV_8UC3 || frame.cols != frame_width_ || frame.rows != frame_height_) {
//         throw std::runtime_error("Invalid input frame: expected " + std::to_string(frame_width_) + "x" +
//                                  std::to_string(frame_height_) + " CV_8UC3");
//     }

//     cv::Rect roi(0, roi_sy_, frame_width_, roi_ey_ - roi_sy_); // 640x108
//     cv::Mat cropped = frame(roi);

//     cv::Mat gray;
//     cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
//     cv::Scalar mean_intensity = cv::mean(gray);
//     float brightness = mean_intensity[0];

//     cv::Mat enhanced;
//     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
//     clahe->setClipLimit(brightness < 100 ? 4.0 : 2.0);
//     clahe->setTilesGridSize(cv::Size(8, 8));
//     if (brightness < 150) {
//         cv::Mat lab;
//         cv::cvtColor(cropped, lab, cv::COLOR_BGR2Lab);
//         std::vector<cv::Mat> lab_channels;
//         cv::split(lab, lab_channels);
//         clahe->apply(lab_channels[0], lab_channels[0]);
//         cv::merge(lab_channels, lab);
//         cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
//     } else {
//         enhanced = cropped;
//     }

//     cv::Mat resized;
//     cv::resize(enhanced, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

//     cv::Mat normalized;
//     resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

//     std::vector<cv::Mat> channels(3);
//     cv::split(normalized, channels);
//     for (int c = 0; c < 3; ++c) {
//         float* dst = input_data_.data() + c * input_height_ * input_width_;
//         memcpy(dst, channels[c].ptr<float>(), input_width_ * input_height_ * sizeof(float));
//     }
// }


// yellow detected ??
// void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
//     preprocess(frame);
//     infer();

//     lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
//     cv::Mat exp_mask;
//     cv::exp(-lane_mask_, exp_mask);
//     lane_mask_ = 1.0 / (1.0 + exp_mask);

//     // Adjust threshold based on brightness
//     cv::Mat gray;
//     cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//     cv::Scalar mean_intensity = cv::mean(gray);
//     float brightness = mean_intensity[0];
//     float threshold = brightness < 100 ? 0.2 : 0.4; // Lowered thresholds for yellow lanes

//     // Use adaptive thresholding for better robustness
//     cv::Mat binary_mask;
//     cv::adaptiveThreshold(lane_mask_, binary_mask, 1.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, -0.2);

//     // Morphological closing with larger kernel
//     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); // Increased kernel size
//     cv::morphologyEx(binary_mask, lane_mask_, cv::MORPH_CLOSE, kernel);

//     cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_NEAREST);

//     output_frame = frame.clone();

//     if (!calculateLaneGeometry(offset, angle)) {
//         std::cout << "[" << __func__ << "] "
//                   << "Failed to calculate lane geometry" << std::endl;
//     }

//     // Use Debug class for visualization
//     debug_->showOutputVideo(output_frame, left_edges_, right_edges_, offset, angle, lane_mask_, visualize_mask);
// }

// yellow not detected
void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

    double min_val, max_val;
    cv::minMaxLoc(lane_mask_, &min_val, &max_val);
    // std::cout << "[processFrame] : lane_mask_ min: " << min_val << ", max: " << max_val << std::endl;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];
    float threshold = brightness < 100 ? 0.15 : 0.3; // Even lower for yellow
    // std::cout << "[processFrame] : Brightness: " << brightness << ", Threshold: " << threshold << std::endl;

    cv::Mat binary_mask;
    cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); // Larger kernel
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_DILATE, kernel);
    cv::morphologyEx(binary_mask, lane_mask_, cv::MORPH_CLOSE, kernel);

    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_NEAREST);

    output_frame = frame.clone();

    if (!calculateLaneGeometry(offset, angle)) {
        std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
    }

    debug_->showOutputVideo(output_frame, left_edges_, right_edges_, offset, angle, lane_mask_, visualize_mask);

    cv::imwrite("lane_mask.png", lane_mask_ * 255);
    cv::imwrite("binary_mask.png", binary_mask * 255);
}