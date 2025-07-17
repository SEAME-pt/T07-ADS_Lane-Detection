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

    // Step 2: Find left and right lane edges using dense sampling
    if (!findLaneEdges(lane_mask_, roi)) {
		std::cerr << "Not enough edge points detected in ROI!" << std::endl;
        return false; // Not enough edge points detected
    }

    // Step 3: Perform weighted linear regression to fit lines to edges
    weightedLinearRegression(left_edges_, iGeo_.left_slope, iGeo_.left_intercept);
    weightedLinearRegression(right_edges_, iGeo_.right_slope, iGeo_.right_intercept);

    // Step 4: Calculate offset and angle from the fitted lines
    float measured_offset, measured_angle;
	calculateOffsetAndAngle(measured_offset, measured_angle);

    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;

	iGeo_.angle = angle * 180 / CV_PI; // Store angle in imgGeometry
	iGeo_.offset = offset ; // Store offset in imgGeometry

    return true;
}

void LaneDetector::defineROI() {
		std::cout << "Defining ROI..." << std::endl;
		roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT); // 252 for 360
		roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);     // 360
		roi_sx_ = ROI_X_BORDER; // 20
		roi_ex_ = frame_width_ - ROI_X_BORDER; // 620
		roi_w_ = roi_ex_ - roi_sx_; // 620 - 20 = 600
		roi_h_ = roi_ey_ - roi_sy_; // 360 - 252 = 108
}

bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) {
    left_edges_.clear();
    right_edges_.clear();

    // STEP 1: find edges on both sides of image
	for (int y = roi.y; y < roi.y + roi.height; ++y) {
        int left_x = -1, right_x = -1;
        for (int x = roi.width / 2; x >= roi.x + ROI_X_BORDER; --x) {
            if (lane_mask.at<float>(y, x) > THRESHOLD) {
                left_x = x;
				left_edges_.emplace_back(left_x, y);
                break;
            }
        }
        for (int x = roi.width / 2; x <= roi.width - ROI_X_BORDER; ++x) {
            if (lane_mask.at<float>(y, x) > THRESHOLD) {
                right_x = x;
				right_edges_.emplace_back(right_x, y);
                break;
            }
        }
    }

	// STEP 2 : Check if we found enough edges
	if (left_edges_.size() < MIN_EDGE_POINTS || right_edges_.size() < MIN_EDGE_POINTS) {
		std::cerr << "[" << __func__ << "] : "
				  << "Not enough edge points found in ROI!" << std::endl;
		return false;
	}

    return !left_edges_.empty() && !right_edges_.empty();
}

void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges,
                                            double& slope, double& intercept) {
    if (edges.size() < MIN_EDGE_POINTS) {
        slope = 0.0;
        intercept = frame_width_ / 2.0 - CAMERA_OFFSET; // 320
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int y_top = static_cast<int>(edges[0].y); // 252
    int y_bottom =static_cast<int>(current_y_top_ + edges.size());     // 360
    int y_range = y_bottom - y_top;

    for (const auto& pt : edges) {
        double y = pt.y;
        double x = pt.x;
        double weight = (y - y_top) / y_range;
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

void LaneDetector::calculateOffsetAndAngle(float& offset, float& angle) const {
	// All values in pixels
    // STEP 1: Define car longitudinal axis = dash cam center position with somer calibration offset
	int xc = frame_width_ / 2 + CAMERA_OFFSET; // Camera center adjusted by offset
    // STEP 2: Calculate middlane offset point xmb at bottom of image, in PIXELS
    int xlb = iGeo_.left_slope * frame_height_ + iGeo_.left_intercept ;
    int xrb = iGeo_.right_slope * frame_height_ + iGeo_.right_intercept;
    int xmb = xc - (xlb + xrb) / 2; // Midpoint distance to car center at bottom

    // STEP 3: Calculate middlane and offset point xmt at center of image, in PIXELS
    int xlt = iGeo_.left_slope * (frame_height_ / 2.0) + iGeo_.left_intercept;
    int xrt = iGeo_.right_slope * (frame_height_ / 2.0) + iGeo_.right_intercept;
	int xmt = xc - (xlt + xrt) / 2;

	// STEP 4 : Convert image midlane offset points [PIXELS] to image Frame coordinate system [METERS]
	// using the equation d[meters] = s(y[pixels]) * x[pixels], where s(y) = Asy * y + Bsy
	// Convert Point at center of image
	double xmt_imgFrame = (Asy * (frame_height_ / 2.0) + Bsy) * xmt;
	// Convert Point at bottom of image
	double xmb_imgFrame = (Asy * (frame_height_) + Bsy) * xmb;

	// STEP 5 : Convert image Frame to car Frame
	// x (image Frame) maps into y car Frame
	// y image Frame maps into x car Frame

	// STEP 5.1 : CENTER point: (xmt_carFrame, ymt_carFrame)
	// img Frame :
	// 		y = frame_height_ / 2 ;
	// 		x = xmt_imgFrame
	// car Frame :
	// 		y = xmt_imgFrame;
	// 		x = calibration point, in meters, distance between car's center of mass and center of dash cam in the ground
	double ymt_carFrame = -xmt_imgFrame;
	double xmt_carFrame = X_IMG_ROI_TOP_CAR_FRAME;
	// STEP 5.2 : BOTTOM point(xmt_carFrame, ymt_carFrame)
	// img Frame :
	// 		y = roi_sy_ + current_y_range_ ;
	// 		x = xmb_imgFrame
	// car Frame :
	// 		y = xmb_imgFrame ;
	// 		x = calibration point, in meters, distance between car's center of mass and bottom of dash cam in the ground
	double xmb_carFrame = X_IMG_ROI_BOTTOM_CAR_FRAME;
	double ymb_carFrame = -xmb_imgFrame;

	// STEP 6 : Calculate car Frame lane geometry
	// Calculate the slope of the middle lane line, in the car Frame
	double slope_carFrame = (ymt_carFrame - ymb_carFrame) / (xmt_carFrame - xmb_carFrame);
	// Calculate the intersect of the middle lane line at the center of mass in the car Frame, this is the ey or offset
	double intercept_carFrame = ymt_carFrame - slope_carFrame * xmt_carFrame;
	// Calculate the yaw angle
	double yaw_angle = std::atan(slope_carFrame); // in radians

	// STEP 7 : prepare vars to be returned
	angle = static_cast<float>(yaw_angle); // Set the angle in radians
	offset = static_cast<float>(intercept_carFrame); // Set the offset in pmeters
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
    cv::resize(frame, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC); // Interpolação cúbica

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB); // Converte de BGR para RGB

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0); // Normaliza para [0,1]

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels); // Canais na ordem R, G, B
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}


void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());

    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

	cv::Mat rawLane;
	cv::threshold(lane_mask_, rawLane, 0.5, 255.0, cv::THRESH_BINARY);
	rawLane.convertTo(rawLane, CV_8U);
	if (!rawLane.empty()) {
		cv::imwrite("rawLane.png", rawLane );
	}

	// fixed threshold, equivalent to (preds > 0.5).float()
	// This threshold can be adjusted based on the model's output characteristics
	// For example, if the model outputs probabilities, a threshold of 0.5 is common.
	// If the model outputs logits, you might need to apply a sigmoid function first.
	// Here we assume lane_mask_ is already in the range [0, 1] after sigmoid activation.
	//
	float threshold = 0.5;
    // cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);

    cv::Mat binary_mask;
    cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);
    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_CUBIC); // Resize to original frame size

    output_frame = frame.clone();

    if (!calculateLaneGeometry(offset, angle)) {
        std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
    }

    debug_->showOutputVideo(rawLane, output_frame, iGeo_, CAMERA_OFFSET);
}