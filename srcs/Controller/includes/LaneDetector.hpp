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


#define ROI_SY_PERCENT 0.5f // ROI starts at 50% of image height
#define ROI_EY_PERCENT 0.95f   // ROI ends at 80% of image height
#define ROI_X_BORDER 0 // Pixels from the left and right edges to avoid noise
#define THRESHOLD 0.5f // Threshold for binary mask
#define MIN_EDGE_POINTS 10 // Coefficient for angle calculation
#define MAX_SEARCH_DISTANCE 310	// Max distance (pixels) to search for edges
#define I_W 256 // Inference image with
#define I_H 128 // Inference image height
#define F_W 640 // Frame width
#define F_H 360 // Frame height

#define KALMAN false // Use Kalman filter for smoothing offset and angle
#define CAR_CM false // Use car center of mass for calculations
// Car frame coordinates : dash cam
#define X_CAR_FRAME_CENTER 0.41f // X coordinate of the car center in the image frame
#define X_CAR_FRAME_BOTTOM 0.20f // X coordinate of the bottom ROI in the image frame
#define CAMERA_Y_POS 0.0f	 // 0 cm in meters, centered on the car's CM
#define CAMERA_X_POS 0.09f	// 9 cm in meters, forward of the car's CM
#define CAMERA_Z_POS 0.115   // 11.5 cm in meters
#define CAMERA_OFFSET 0 // Offset in pixels, adjust if needed
#define CAMERA_TILT 19.0 * CV_PI / 180.0 // 19 degrees in radians (19 * pi/180)

// Fixed parameters as constants
#define CAMERA_FOCAL_LENGTH = 0.00315 // Focal length in meters (2 mm)

//New members for lane geometry historical data
#define MAX_HISTORY_SIZE 10 // Maximum size of the history

// Coefficients for distance calculation, converting pixels to meters
// Equations :
//  d(m) = s(y) * x
// 	s(y) = a * y + b
// x and y are pixel coordinates
#define Asy -2.6e-6 // Coefficient for distance calculation
#define Bsy 1.35e-3 // Coefficient for distance calculation

typedef struct s_carFrame {
	float xT{X_CAR_FRAME_CENTER};	// X coordinate
	float yT{0.0f};	// Intercept of the left lane line
	float xB{X_CAR_FRAME_BOTTOM};	// Slope of the right lane line
	float yB{0.0f};	// Intercept of the right lane line
	float xDelta{X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM};	// Delta X between top and bottom points
	float slope{0.0f};	// Slope of the right lane line
	float intercept{0.0f};	// Intercept of the right lane line
	float angle{0.0f};	// Angle of the lane in radians
} t_carFrame;

typedef struct s_imgFrame {
	int xltPX{0};	// Left edge at top
	int xrtPX{0};	// Right edge at top
	int xlbPX{0};	// Left edge at bottom
	int xrbPX{0};	// Right edge at bottom
	int xmtPX{0};	// Midpoint at top
	int xmbPX{0};	// Midpoint at bottom
	int xcPX{F_W / 2 - CAMERA_OFFSET};	// Center of the image
	float xmt{0.0f};	// Midpoint at top in meters
	float xmb{0.0f};	// Midpoint at bottom in meters
} t_imgFrame;

enum KalmanStateIndex { OFFSET = 0, OFFSET_VEL = 1, ANGLE = 2 };
enum KalmanMeasurementIndex { MEASUREMENT_OFFSET = 0, MEASUREMENT_ANGLE = 1 };
enum KalmanPredictionIndex { PREDICTION_OFFSET = 0, PREDICTION_ANGLE = 1 };
enum KalmanPredictionCovIndex { PREDICTION_COV_OFFSET = 0, PREDICTION_COV_ANGLE = 1 };
enum KalmanMeasurementCovIndex { MEASUREMENT_COV_OFFSET = 0, MEASUREMENT_COV_ANGLE = 1 };
enum KalmanErrorCovIndex { ERROR_COV_OFFSET = 0, ERROR_COV_ANGLE = 1 };
enum KalmanProcessCovIndex { PROCESS_COV_OFFSET = 0, PROCESS_COV_ANGLE = 1 };
enum KalmanTransitionIndex { TRANSITION_OFFSET = 0, TRANSITION_VEL = 1, TRANSITION_ANGLE = 2 };
enum KalmanMeasurementMatrixIndex { MEASUREMENT_MATRIX_OFFSET = 0, MEASUREMENT_MATRIX_ANGLE = 1 };


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


	// Prediction
	// Kalman Filter
	// cv::KalmanFilter kf_;		 // Kalman filter for smoothing offset and angle
	// float offset_kalman_{0.0f};
	// float angle_kalman_{0.0f};


	// cv::Mat measurement_;
	// cv::Mat prediction_;


	// image vertical useful range of the edges
	float current_y_top_ = 0.0f; // Y-coordinate of the top of the ROI
	float current_y_bottom_ = 0.0f; // Y-coordinate of the bottom of the ROI
	float current_y_range_ = 0.0f;// = current_y_bottom_ - current_y_top_; // Range of Y-coordinates in the ROI

};

#endif // LANE_DETECTOR_HPP