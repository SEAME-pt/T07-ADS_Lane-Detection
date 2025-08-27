#include "LaneDetector.hpp"
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
	cudaStreamCreate(&stream_);

	    // Initialize Kalman filter
	if (KALMAN) {
		kf_ = cv::KalmanFilter(2, 2, 0, CV_32F);
		kf_.statePre.at<float>(0) = 0.0f;
		kf_.statePre.at<float>(1) = 0.0f;
		kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
		kf_.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
		cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-4));
		cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
		cv::setIdentity(kf_.errorCovPre, cv::Scalar::all(1));

		    offset_kalman_ = 0.0f;
    	angle_kalman_ = 0.0f;
    	estimated_lane_width_ = 200.0f;
	}



	input_height_ = 128;
	input_width_ = 256;
	roi_sy_ = static_cast<int>(F_H * ROI_SY_PERCENT);
	roi_ey_ = static_cast<int>(F_H * ROI_EY_PERCENT);
	roi_sx_ = ROI_X_BORDER;
	roi_ex_ = F_W - ROI_X_BORDER;
	roi_w_ = roi_ex_ - roi_sx_;
	roi_h_ = roi_ey_ - roi_sy_;

	// low pass filter parameters
    offset_smooth_ = 0.0f;
    angle_smooth_ = 0.0f;
    alpha_ = 0.5f; // Low-pass filter coefficient

	// initialize image frame geometry
	estimated_lane_width_ = -1.0f;
	prev_left_edge_ = F_W / 2;
	prev_right_edge_ = F_W / 2;
	last_left_edge_ = F_W / 2;
	last_right_edge_ = F_W / 2;

	defineROI();
	debug_ = std::make_unique<Debug>(F_W, F_H, roi_sy_, roi_ey_);
	loadEngine(trt_model_path);
	std::cout << "LaneDetector created with model: " << trt_model_path << std::endl;
}

LaneDetector::~LaneDetector() {
	cudaStreamDestroy(stream_);
	cudaFree(buffers_[0]);
	cudaFree(buffers_[1]);
}

/// @brief Calculate lane geometry based on detected edges.
/// This function estimates the offset and angle of the lane based on the detected left and right edges.
/// It uses the slopes and intercepts of the detected edges to compute the lane geometry.
/// @param offset	The offset from the center of the lane in meters, in the car frame system of coordinates.
/// @param angle	The angle of the lane in radians, in the car frame system of coordinates.
/// @return 		True if lane geometry was successfully calculated, false otherwise.
void LaneDetector::defineROI() {
    std::cout << "[" << __func__ << "] Started..." << std::endl;
    roi_sy_ = static_cast<int>(F_H * ROI_SY_PERCENT);
    roi_ey_ = static_cast<int>(F_H * ROI_EY_PERCENT);
    // Dynamic ROI adjustment based on angle in radians
    // int shift = static_cast<int>(iGeo_.angle * 114.6f); // ~2 pixels per degree (1 radian ≈ 57.3 degrees)
    roi_sx_ = std::max(0, ROI_X_BORDER);
    // roi_sx_ = std::max(0, ROI_X_BORDER - shift);
    roi_ex_ = std::min(F_W, F_W - ROI_X_BORDER);
    // roi_ex_ = std::min(F_W, F_W - ROI_X_BORDER - shift);
    roi_w_ = roi_ex_ - roi_sx_;
    roi_h_ = roi_ey_ - roi_sy_;
    std::cout << "[" << __func__ << "] Ready ROI" << std::endl;
}

/// @brief 				Create left and right edges vectors from lane mask.
/// This function scans the lane mask within the specified ROI to find the left and right edges of the lane.
/// It uses a dense sampling approach to identify the first detected lane pixel in each row.
/// @note 				The function clears the left_edges_ and right_edges_ vectors before populating them.
/// @param lane_mask 	The lane mask image containing the detected lane pixels.
/// @param roi			The region of interest (ROI) within the lane mask to search for edges.
/// @return 			True if enough edge points were found, false otherwise.
bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) {

	left_edges_.clear();
	right_edges_.clear();
	int scan_padding = 10; // Padding to avoid edge effects
	bool left_edge_ready, right_edge_ready;

	// scan limits
	int left_scan_start = roi.width / 2 - scan_padding;
	int left_scan_end = roi.x + scan_padding;
	int right_scan_start = roi.width / 2 + scan_padding;
	int right_scan_end = (roi.width - 1) - scan_padding;

	// Reset edge readiness
	left_edge_ready = false;
	right_edge_ready = false;
	// std::cout << "[" << __func__ << "] : Scanning for lane edges in ROI: " << roi << std::endl;
	// Scan from bottom to top of the ROI
	for (int y = roi.y + roi.height; y > roi.y; --y) {
		// Reset edge readiness for each row
		int left_x = -1, right_x = -1;

		for (int x = left_scan_start; x >= left_scan_end; --x) {
			if (lane_mask.at<float>(y, x) > LANE_THRESHOLD && !left_edge_ready) {
				// Update left_scan_start for next row
				left_scan_start = std::min(x + scan_padding, roi.width / 2 - scan_padding); // Update start_x_sweep for next row
				left_scan_end = std::max(x - scan_padding, 0);
				left_x = x;
				left_edges_.emplace_back(left_x, y);
				if (left_scan_start == roi.width / 2 - scan_padding) {
					// If we already found a left edge, skip further checks
					left_edge_ready = true;
				}
				break;
			}
		}

		for (int x = right_scan_start; x <= right_scan_end; ++x) {
			if (lane_mask.at<float>(y, x) > LANE_THRESHOLD && !right_edge_ready) {
				right_scan_start = std::max(x - scan_padding, roi.width / 2 + scan_padding); // Update right_scan_start for next row
				right_scan_end = std::min(x + scan_padding, roi.width - 1);
				right_x = x;
				right_edges_.emplace_back(right_x, y);
				if (right_scan_start == roi.width / 2 + scan_padding) {
					// If we already found a right edge, skip further checks
					right_edge_ready = true;
				}
				break;
			}
		}

	}
	// std::cout << "[" << __func__ << "] : Left edges found: " << left_edges_.size() << ", Right edges found: " << right_edges_.size() << std::endl;
	return left_edges_.size() >= MIN_EDGE_POINTS && right_edges_.size() >= MIN_EDGE_POINTS;
}

/// @brief 	Calculate the weighted linear regression for the detected edges.
/// This function computes the slope and intercept of the line that best fits
/// the detected edges using a weighted linear regression.
/// @note 	The function uses a weighted approach where the weight is based
/// on the vertical position of the edge points.
/// @param edges		The vector of edge points detected in the lane mask.
/// @param slope		The slope of the fitted line.
/// @param intercept 	The intercept of the fitted line.
void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges, double& slope, double& intercept) {
	if (edges.size() < MIN_EDGE_POINTS) {
		slope = 0.0;
		intercept = F_W / 2.0 - CAMERA_OFFSET;
		return;
	}

	double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
	int y_start = static_cast<int>(edges[0].y); // 252
	int y_end =static_cast<int>(y_start + edges.size());
	int y_range = y_end - y_start;

	for (const auto& pt : edges) {
		double y = pt.y;
		double x = pt.x;
		double weight = (y - y_start) / y_range;
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

// Helper
float LaneDetector::smoothValue(std::deque<float>& history, float new_value) {
	history.push_back(new_value);
	if (history.size() > MISSING_EDGE_HISTORY_SIZE) {
		history.pop_front();
	}
	return std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
}


/// @brief Calculate the middle lane line based on the detected edges.
/// This function calculates the left and right edges of the lane at the top and bottom of the
/// image frame, and computes the midpoints at the top and bottom.
/// It uses the slopes and intercepts of the detected edges to compute the midpoints.
/// It also calculates the lane width based on the detected edges.
/// @note This function assumes that the left and right edges have been detected and stored in the
/// `left_edges_` and `right_edges_` vectors.
/// It updates the `imgFrame_` structure with the calculated midpoints and lane width.
/// If any edge is missing, it estimates the missing edge based on the available edge and the estimated lane width.
/// @return 	True if the middle lane line was successfully calculated, false otherwise.
void LaneDetector::calculateMiddleLaneLine(void) {
	bool left_valid = left_edges_.size() >= MIN_EDGE_POINTS;
	bool right_valid = right_edges_.size() >= MIN_EDGE_POINTS;

	float lane_width_meters = estimated_lane_width_ < 0.0f ? 0.25f : estimated_lane_width_;
	iGeo_.lane_width = lane_width_meters;

	if (left_valid) {
		imgFrame_.xltPX = iGeo_.left_slope * (F_H / 2.0f) + iGeo_.left_intercept;
		imgFrame_.xlbPX = iGeo_.left_slope * F_H + iGeo_.left_intercept;
	}
	if (right_valid) {
		imgFrame_.xrtPX = iGeo_.right_slope * (F_H / 2.0f) + iGeo_.right_intercept;
		imgFrame_.xrbPX = iGeo_.right_slope * F_H + iGeo_.right_intercept;
	}

	if (left_valid && right_valid) {
		float lane_width_new = ((Asy * F_H + Bsy) * (imgFrame_.xrbPX - imgFrame_.xlbPX)
			+ (Asy * (F_H / 2.0f) + Bsy) * (imgFrame_.xrtPX - imgFrame_.xltPX)) / 2.0f;
		lane_width_history_.push_back(lane_width_new);
		if (lane_width_history_.size() > MAX_HISTORY_SIZE) {
			lane_width_history_.erase(lane_width_history_.begin());
		}
		estimated_lane_width_ = std::accumulate(lane_width_history_.begin(), lane_width_history_.end(), 0.0f) / lane_width_history_.size();
		iGeo_.lane_width = estimated_lane_width_;
	}
	else if (left_valid && !right_valid) {
		float right_top = iGeo_.lane_width / (Asy * (F_H / 2.0f) + Bsy) + imgFrame_.xltPX;
		float right_bottom = iGeo_.lane_width / (Asy * F_H + Bsy) + imgFrame_.xlbPX;

		imgFrame_.xrtPX = smoothValue(missing_right_top_history_, right_top);
		imgFrame_.xrbPX = smoothValue(missing_right_bottom_history_, right_bottom);

		iGeo_.right_slope = (imgFrame_.xrtPX - imgFrame_.xrbPX) / ((F_H / 2.0f) - F_H);
		iGeo_.right_intercept = imgFrame_.xrbPX - iGeo_.right_slope * F_H;
	}
	else if (!left_valid && right_valid) {
		float left_top = imgFrame_.xrtPX - iGeo_.lane_width / (Asy * (F_H / 2.0f) + Bsy);
		float left_bottom = imgFrame_.xrbPX - iGeo_.lane_width / (Asy * F_H + Bsy);

		imgFrame_.xltPX = smoothValue(missing_left_top_history_, left_top);
		imgFrame_.xlbPX = smoothValue(missing_left_bottom_history_, left_bottom);

		iGeo_.left_slope = (imgFrame_.xltPX - imgFrame_.xlbPX) / ((F_H / 2.0f) - F_H);
		iGeo_.left_intercept = imgFrame_.xlbPX - iGeo_.left_slope * F_H;
	}

	imgFrame_.xmtPX = imgFrame_.xcPX - (imgFrame_.xltPX + imgFrame_.xrtPX) / 2.0f;
	imgFrame_.xmbPX = imgFrame_.xcPX - (imgFrame_.xlbPX + imgFrame_.xrbPX) / 2.0f;

	imgFrame_.xmt = (Asy * (F_H / 2.0f) + Bsy) * imgFrame_.xmtPX;
	imgFrame_.xmb = (Asy * F_H + Bsy) * imgFrame_.xmbPX;

}

/// @brief Calculate the offset and angle of the lane in the car frame.
/// This function converts the image frame midlane points from pixels to meters and calculates the offset and
/// angle of the lane in the car frame system of coordinates.
/// The offset is the distance from the center of the car to the lane, and the angle
/// is the yaw angle of the lane in radians.
/// The conversion uses a scale function defined by the slope (Asy) and intercept (Bsy) of the scale function.
/// @param offset	The offset from the center of the car to the lane in meters.
/// @param angle	The yaw angle of the lane in radians.
void LaneDetector::calculateOffsetAndAngle(float& offset, float& angle) {

	carFrame_.yT = -imgFrame_.xmt;
	carFrame_.yB = -imgFrame_.xmb; // ymb_carFrame is the bottom point in the car Frame
	carFrame_.slope = (carFrame_.yT - carFrame_.yB) / (carFrame_.xDelta);
	carFrame_.intercept = carFrame_.yT - carFrame_.slope * carFrame_.xT;

	if (CAR_CM) {
		offset = carFrame_.intercept;
	} else {
		float ey = carFrame_.slope * X_CAR_FRAME_BOTTOM + carFrame_.intercept;
		offset = static_cast<float>(ey);
	}
	angle = static_cast<float>(std::atan(carFrame_.slope));
}

/// @brief Calculate lane geometry based on detected edges.
/// This function estimates the offset and angle of the lane based on the detected left and right edges.
/// It uses the slopes and intercepts of the detected edges to compute the lane geometry.
/// @param offset	The offset from the center of the lane in meters, in the car frame system of coordinates.
/// @param angle	The angle of the lane in radians, in the car frame system of coordinates.
/// @return 		True if lane geometry was successfully calculated, false otherwise.
bool LaneDetector::calculateLaneGeometry(float& offset, float& angle, bool visualize_mask) {
	if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
		std::cerr << "Invalid lane mask!" << std::endl;
		return false;
	}

	cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);
	if (roi.width <= 0 || roi.height <= 0) {
		std::cerr << "Invalid ROI dimensions!" << std::endl;
		return false;
	}

	if (!findLaneEdges(lane_mask_, roi)) {
		if (KALMAN) {
			float smoothed_offset, smoothed_angle;
			applyKalmanFilter(iGeo_.offset , iGeo_.angle, smoothed_offset, smoothed_angle);
			offset = smoothed_offset;
			angle = smoothed_angle;
			iGeo_.offset = smoothed_offset;;
			iGeo_.angle = smoothed_angle;
		} else {
			offset = offset_smooth_;
			angle = angle_smooth_;
			iGeo_.offset = offset;
			iGeo_.angle = angle;
    	}
		return true;
	}

	if (left_edges_.size() >= MIN_EDGE_POINTS)
		weightedLinearRegression(left_edges_, iGeo_.left_slope, iGeo_.left_intercept);
	if (right_edges_.size() >= MIN_EDGE_POINTS)
		weightedLinearRegression(right_edges_, iGeo_.right_slope, iGeo_.right_intercept);

	float measured_offset, measured_angle;
	calculateMiddleLaneLine();
	calculateOffsetAndAngle(measured_offset, measured_angle);

	if (KALMAN) {
		float smoothed_offset, smoothed_angle;
		applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);
		offset = smoothed_offset;
		angle = smoothed_angle;
	} else {
		offset_smooth_ = alpha_ * measured_offset + (1.0f - alpha_) * offset_smooth_;
		angle_smooth_ = alpha_ * measured_angle + (1.0f - alpha_) * angle_smooth_;
		offset = offset_smooth_;
		angle = angle_smooth_;
	}

	iGeo_.angle = angle; // Store angle in imgGeometry
	iGeo_.offset = offset ; // Store offset in imgGeometry
	history_.push_back(iGeo_);
	if (history_.size() > MAX_HISTORY_SIZE) {
		history_.erase(history_.begin());
	}
	return true;
}

void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle,
									 float& smoothed_offset, float& smoothed_angle) {
	cv::Mat prediction = kf_.predict();
	cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured_offset, measured_angle);
	cv::Mat corrected = kf_.correct(measurement);
	smoothed_offset = corrected.at<float>(0);
	smoothed_angle = corrected.at<float>(1);
}

/// @brief Load the TensorRT engine from a file.
/// @param trt_model_path	The path to the TensorRT model file.
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

	cudaError_t err = cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
	if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
	err = cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
	if (err != cudaSuccess) {
		cudaFree(buffers_[0]);
		throw std::runtime_error("CUDA malloc failed for output buffer: " + std::string(cudaGetErrorString(err)));
	}

	input_data_.resize(1 * 3 * input_height_ * input_width_);
	output_data_.resize(1 * 1 * input_height_ * input_width_);
}

/// @brief Run inference on the preprocessed input data.
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

/// @brief Preprocess the input frame for model inference.
/// This function resizes the input frame to the model's expected input size,
/// converts the color space from BGR to RGB, normalizes the pixel values,
/// and prepares the input data in a contiguous format suitable for TensorRT inference.
/// @param frame	The input video frame to be preprocessed.
void LaneDetector::preprocess(const cv::Mat& frame) {
	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC);

	cv::Mat rgb;
	cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

	rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

	std::vector<cv::Mat> channels;
	cv::split(rgb, channels);
	for (int c = 0; c < 3; ++c) {
		memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
	}
}

/// @brief Process a video frame to detect lanes and calculate geometry.
/// This function processes a single video frame to detect lanes using a deep learning model,
/// calculates the lane geometry (offset and angle), and optionally visualizes the lane mask.
/// @param frame	The input video frame to be processed.
/// @param offset	The calculated offset from the center of the lane in meters.
/// @param angle	The calculated angle of the lane in radians.
/// @param output_frame	The output frame with visualizations (if enabled).
/// @param visualize_mask	Flag to enable visualization of the lane mask.
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

	float threshold = LANE_THRESHOLD;

	cv::Mat binary_mask;
	cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);
	cv::resize(lane_mask_, lane_mask_, cv::Size(F_W, F_H), 0, 0, cv::INTER_CUBIC);

	output_frame = frame.clone();

	bool laneOk = calculateLaneGeometry(offset, angle, visualize_mask);
	if (!laneOk) {
		std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
	}

	debug_->showOutputVideo(rawLane, output_frame, iGeo_, CAMERA_OFFSET);
}