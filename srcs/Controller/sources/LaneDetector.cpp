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

	// std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
    //                        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    //                        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
	std::string pipeline = "nvarguscamerasrc exposuretimerange=\"1000000 50000000\" gainrange=\"1 16\" !"
	                       "video/x-raw(memory:NVMM), width=640, height=360, "
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
	if (roi.width <= 0 || roi.height <= 0) {
		std::cerr << "Invalid ROI dimensions!" << std::endl;
		return false; // Invalid ROI dimensions
	}
	// std::cout << "[" << __func__ << "] : ROI : \n"
	// 		  << "roi.x = " << roi.x << std::endl
	// 		  << "roi.y = " << roi.y << std::endl
	// 		  << "roi.width = " << roi.width << std::endl
	// 		  << "roi.height = " << roi.height << std::endl;


    // Step 2: Find left and right lane edges using dense sampling
    // std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi)) {
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
	// std::cout << "[" << __func__ << "] "
	// 		  << "Measured Offset: " << std::setw(6) << measured_offset
	// 		  << " m, Measured Angle: " << std::setw(6) << measured_angle << " rad"
	// 		  << '\r' << std::flush;


	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;
    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;
	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;

	// std::cout << "[" << __func__ << "] "
	// 		  << "Offset: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_offset
	// 		  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << offset << ") m"
	// 		  << "Angle: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_angle
	// 		  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << smoothed_angle << ") rad"
	// 		  << '\r' << std::flush;


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

bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) {
    left_edges_.clear();
    right_edges_.clear();

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        int left_x = -1, right_x = -1;
		// std::cout << "[" << __func__ << "] "
		// 		  << "Searching for left edges at y = " << y
		// 		  << " in ROI: [ from x = " << roi.width /2 << " to x >= " << roi.x + MAX_SEARCH_DISTANCE << " ]"
		// 		  << std::endl;
        for (int x = roi.width /2; x >= roi.x + ROI_X_BORDER; --x) {
            if (lane_mask.at<float>(y, x) > 0.5f) {
                left_x = x;
				// std::cout << "Left edge found at y = " << y << ", x = " << left_x << std::endl;
                break;
            }
        }
		// std::cout << "[" << __func__ << "] "
		// 		  << "Searching for right edges at y = " << y
		// 		  << " in ROI: [ from x = " << roi.width /2 << " to x <= " << roi.x + MAX_SEARCH_DISTANCE << " ]"
		// 		  << std::endl;
        for (int x = roi.width/2; x <= roi.width - ROI_X_BORDER; ++x) {
            if (lane_mask.at<float>(y, x) > 0.5f) {
                right_x = x;
				// std::cout << "Right edge found at y = " << y << ", x = " << right_x << std::endl;
                break;
            }
        }
        if (left_x != -1 && right_x != -1 && left_x < right_x) {
            left_edges_.emplace_back(left_x, y);
            right_edges_.emplace_back(right_x, y);
        }
    }
	std::cout << "[" << __func__ << "] : "
				<< "Found " << left_edges_.size() << " left edges and "
				<< right_edges_.size() << " right edges in the ROI." << std::endl;
    return !left_edges_.empty() && !right_edges_.empty();
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

void LaneDetector::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_)); // resize to input size

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // Normaliza para [0,1]

    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

// yellow NOT processFrame():
void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

    double min_val, max_val;
    cv::minMaxLoc(lane_mask_, &min_val, &max_val);
    std::cout << "["<<__func__<< "] : lane_mask_ min: " << min_val << ", max: " << max_val << std::endl;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_intensity = cv::mean(gray);
    float brightness = mean_intensity[0];
    float threshold = brightness < 100 ? 0.1 : 0.3; // Even lower for yellow
    std::cout << "["<<__func__<< "] : Brightness: " << brightness << ", Threshold: " << threshold << std::endl;

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