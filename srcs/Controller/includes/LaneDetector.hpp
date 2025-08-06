#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "Debug.hpp"
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cmath>

#include "Configs.hpp"

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) std::cerr << msg << std::endl;
	}
};

class LaneDetector {
public:
	LaneDetector(const std::string& trt_model_path);
	~LaneDetector();
	void processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask);


private:

	void loadEngine(const std::string& trt_model_path);
	void preprocess(const cv::Mat& frame);
	void infer();
	void defineROI() ;
	bool calculateLaneGeometry(float& offset, float& angle, bool visualize_mask);
	bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) ;
	void weightedLinearRegression(const std::vector<cv::Point>& points, double& slope, double& intercept) ;
	void calculateMiddleLaneLine(void) ;
	void calculateOffsetAndAngle(float& offset, float& angle);
	void applyKalmanFilter(float measured_offset, float measured_angle, float& smoothed_offset, float& smoothed_angle);

	// Kalman Filter
	cv::KalmanFilter kf_;		 // Kalman filter for smoothing offset and angle
	float offset_kalman_{0.0f};
	float angle_kalman_{0.0f};

	cv::Mat measurement_;
	cv::Mat prediction_;

	std::vector<imgGeometry> history_; // History of lane geometry parameters
	std::vector<float> lane_width_history_; // This is used to store the last known lane geometry width
	float estimated_lane_width_{-1.0f}; // when one lane edge is missing
    float offset_smooth_, angle_smooth_;
    float alpha_; // Low-pass filter coefficient
	// Dimensions
	int input_height_{I_H};
	int input_width_{I_W};
	int frame_height_{F_H};
	int frame_width_{F_W};
	// ROI
	int roi_sx_{20};
	int roi_sy_{252};
	int roi_ex_{620};
	int roi_ey_{360};
	int roi_w_{600};
	int roi_h_{108};
	// Historical data
	int prev_left_edge_{320};
	int prev_right_edge_{320};
	int last_left_edge_pix_{320};
	int last_right_edge_pix_{320};
	float last_left_edge_ = -1.0f;  // Store last known left edge position
	float last_right_edge_ = -1.0f; // Store last known right edge position
	cv::Mat lane_mask_;
	std::vector<cv::Point> left_edges_, right_edges_;

	// OpenCV
	// TensorRT
	cudaStream_t stream_;
	std::vector<float> input_data_, output_data_;
	void* buffers_[2];
	std::unique_ptr<nvinfer1::IRuntime> runtime_;
	std::unique_ptr<nvinfer1::ICudaEngine> engine_;
	std::unique_ptr<nvinfer1::IExecutionContext> context_;
	Logger logger_;
	// Debugging
	std::unique_ptr<Debug> debug_;
	imgGeometry iGeo_; // Structure to hold lane geometry parameters
	t_carFrame carFrame_;
	t_imgFrame imgFrame_;

	// image vertical useful range of the edges
	float current_y_top_ = 0.0f; // Y-coordinate of the top of the ROI
	float current_y_bottom_ = 0.0f; // Y-coordinate of the bottom of the ROI
	float current_y_range_ = 0.0f;// = current_y_bottom_ - current_y_top_; // Range of Y-coordinates in the ROI

};

#endif // LANE_DETECTOR_HPP