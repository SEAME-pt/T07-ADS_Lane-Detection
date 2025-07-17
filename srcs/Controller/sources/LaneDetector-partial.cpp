#include "LaneDetector.hpp"
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
	cudaStreamCreate(&stream_);
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
	frame_height_ = 360;
	frame_width_ = 640;
	roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT);
	roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);
	roi_sx_ = ROI_X_BORDER;
	roi_ex_ = frame_width_ - ROI_X_BORDER;
	roi_w_ = roi_ex_ - roi_sx_;
	roi_h_ = roi_ey_ - roi_sy_;

	offset_kalman_ = 0.0f;
	angle_kalman_ = 0.0f;
	estimated_lane_width_ = 200.0f;
	prev_left_edge_ = frame_width_ / 2;
	prev_right_edge_ = frame_width_ / 2;
	last_left_edge_ = frame_width_ / 2;
	last_right_edge_ = frame_width_ / 2;

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
	std::cout << "[" << __func__ << "] Camera pipeline opened successfully: \n" << pipeline << std::endl;
	std::cout << "[" << __func__ << "] LaneDetector initialization concluded!" << std::endl;
	return cap_.isOpened();
}

void LaneDetector::defineROI() {
	std::cout << "Defining ROI..." << std::endl;
	roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT);
	roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);
	roi_sx_ = ROI_X_BORDER;
	roi_ex_ = frame_width_ - ROI_X_BORDER;
	roi_w_ = roi_ex_ - roi_sx_;
	roi_h_ = roi_ey_ - roi_sy_;
}

bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) {
	left_edges_.clear();
	right_edges_.clear();

	// Find edges in ROI
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

	// Handle missing edges
	bool left_valid = left_edges_.size() >= MIN_EDGE_POINTS;
	bool right_valid = right_edges_.size() >= MIN_EDGE_POINTS;

	if (left_valid && right_valid) {
		// Update lane width estimate
		float lane_width = 0.0f;
		int count = 0;
		for (size_t i = 0; i < std::min(left_edges_.size(), right_edges_.size()); ++i) {
			if (left_edges_[i].y == right_edges_[i].y) {
				lane_width += right_edges_[i].x - left_edges_[i].x;
				count++;
			}
		}
		if (count > 0) {
			lane_width /= count;
			lane_width_history_.push_back(lane_width);
			if (lane_width_history_.size() > max_history_size_) {
				lane_width_history_.erase(lane_width_history_.begin());
			}
			estimated_lane_width_ = std::accumulate(lane_width_history_.begin(), lane_width_history_.end(), 0.0f) / lane_width_history_.size();
		}
		return true;
	} else if (left_valid && !right_valid) {
		// Estimate right edge using left edge and historical lane width
		for (const auto& pt : left_edges_) {
			int right_x = pt.x + static_cast<int>(estimated_lane_width_);
			right_edges_.emplace_back(right_x, pt.y);
		}
		return true;
	} else if (!left_valid && right_valid) {
		// Estimate left edge using right edge and historical lane width
		for (const auto& pt : right_edges_) {
			int left_x = pt.x - static_cast<int>(estimated_lane_width_);
			left_edges_.emplace_back(left_x, pt.y);
		}
		return true;
	} else {
		// Both edges missing, rely on Kalman filter prediction
		return false;
	}
}

void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges, double& slope, double& intercept) {
	if (edges.size() < MIN_EDGE_POINTS) {
		slope = 0.0;
		intercept = frame_width_ / 2.0 - CAMERA_OFFSET;
		return;
	}

	double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
	int y_top = static_cast<int>(edges[0].y);
	int y_bottom = static_cast<int>(edges[0].y + edges.size());
	int y_range = y_bottom - y_top;

	for (const auto& pt : edges) {
		double y = pt.y;
		double x = pt.x;
		double weight = (y - y_top) / static_cast<double>(y_range);
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
	// Calculate middle lane line from fitted edges
	int xc = frame_width_ / 2 + CAMERA_OFFSET;
	int xlb = iGeo_.left_slope * frame_height_ + iGeo_.left_intercept;
	int xrb = iGeo_.right_slope * frame_height_ + iGeo_.right_intercept;
	int xmb = (xlb + xrb) / 2; // Midpoint at bottom
	int xlt = iGeo_.left_slope * (frame_height_ / 2.0) + iGeo_.left_intercept;
	int xrt = iGeo_.right_slope * (frame_height_ / 2.0) + iGeo_.right_intercept;
	int xmt = (xlt + xrt) / 2; // Midpoint at center

	// Convert to image frame (meters)
	double xmt_imgFrame = (Asy * (frame_height_ / 2.0) + Bsy) * (xmt - xc);
	double xmb_imgFrame = (Asy * frame_height_ + Bsy) * (xmb - xc);

	// Convert to car frame
	double ymt_carFrame = -xmt_imgFrame;
	double xmt_carFrame = X_IMG_ROI_TOP_CAR_FRAME;
	double xmb_carFrame = X_IMG_ROI_BOTTOM_CAR_FRAME;
	double ymb_carFrame = -xmb_imgFrame;

	// Calculate middle lane geometry in car frame
	double slope_carFrame = (ymt_carFrame - ymb_carFrame) / (xmt_carFrame - xmb_carFrame);
	double intercept_carFrame = ymt_carFrame - slope_carFrame * xmt_carFrame;
	double yaw_angle = std::atan(slope_carFrame);

	offset = static_cast<float>(intercept_carFrame);
	angle = static_cast<float>(yaw_angle);
}

bool LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
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
		// Use Kalman filter prediction if both edges are missing
		cv::Mat prediction = kf_.predict();
		offset = prediction.at<float>(0);
		angle = prediction.at<float>(1);
		iGeo_.offset = offset;
		iGeo_.angle = angle * 180 / CV_PI;
		return true;
	}

	weightedLinearRegression(left_edges_, iGeo_.left_slope, iGeo_.left_intercept);
	weightedLinearRegression(right_edges_, iGeo_.right_slope, iGeo_.right_intercept);

	float measured_offset, measured_angle;
	calculateOffsetAndAngle(measured_offset, measured_angle);

	float smoothed_offset, smoothed_angle;
	applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

	offset = smoothed_offset;
	angle = smoothed_angle;

	iGeo_.offset = offset;
	iGeo_.angle = angle * 180 / CV_PI;

	// Store geometry in history
	history_.push_back(iGeo_);
	if (history_.size() > max_history_size_) {
		history_.erase(history_.begin());
	}

	return true;
}

// Other methods (unchanged)
void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle, float& smoothed_offset, float& smoothed_angle) {
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

void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
	preprocess(frame);
	infer();
	lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
	cv::Mat exp_mask;
	cv::exp(-lane_mask_, exp_mask);
	lane_mask_ = 1.0 / (1.0 + exp_mask);
	cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_CUBIC);
	output_frame = frame.clone();
	if (!calculateLaneGeometry(offset, angle)) {
		std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
	}
	debug_->showOutputVideo(lane_mask_, output_frame, iGeo_, CAMERA_OFFSET);
}