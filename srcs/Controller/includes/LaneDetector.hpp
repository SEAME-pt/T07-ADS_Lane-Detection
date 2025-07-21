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


static constexpr int ROI_X_BORDER = 0; // Pixels from the left and right edges to avoid noise
static constexpr int I_W = 256;
static constexpr int I_H = 128;
static constexpr int F_W = 640; // Frame width
static constexpr int F_H = 360; // Frame height

static constexpr double X_CAR_FRAME_CENTER = 0.41f; // X coordinate of the car center in the image frame
static constexpr double X_CAR_FRAME_BOTTOM = 0.20f; // X coordinate of the bottom ROI in the image frame

// Fixed parameters as constants
static constexpr double CAMERA_TILT = 19.0 * CV_PI / 180.0; // 19 degrees in radians (19 * pi/180)
static constexpr double CAMERA_X_POS = 0.09f;	// 9 cm in meters, forward of the car's CM
static constexpr double CAMERA_Y_POS = 0.0f;	 // 0 cm in meters, centered on the car's CM
static constexpr double CAMERA_Z_POS = 0.115;   // 11.5 cm in meters
static constexpr double CAMERA_FOCAL_LENGTH = 0.00315; // Focal length in meters (2 mm)
static constexpr int CAMERA_OFFSET = 0; // Offset in pixels, adjust if needed

static constexpr float ROI_SY_PERCENT = 0.5f; // ROI starts at 50% of image height
static constexpr float ROI_EY_PERCENT = 0.9f;   // ROI ends at 80% of image height
static constexpr int MAX_SEARCH_DISTANCE = 310;	// Max distance (pixels) to search for edges
// static constexpr double C_DISTANCE = 0.0001; // Coefficient for distance calculation
static constexpr double MIN_EDGE_POINTS = 10; // Coefficient for angle calculation
static constexpr double THRESHOLD = 0.5; // Threshold for binary mask

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
	// OpenCV
	cv::Mat lane_mask_;
	std::vector<float> input_data_;
	std::vector<float> output_data_;
	void* buffers_[2];
	// TensorRT
	cudaStream_t stream_;
	std::unique_ptr<nvinfer1::IRuntime> runtime_;
	std::unique_ptr<nvinfer1::ICudaEngine> engine_;
	std::unique_ptr<nvinfer1::IExecutionContext> context_;
	// Debugging
	Logger logger_;
	std::unique_ptr<Debug> debug_;
	imgGeometry iGeo_; // Structure to hold lane geometry parameters
	t_carFrame carFrame_;
	t_imgFrame imgFrame_;
	std::vector<cv::Point> left_edges_;
	std::vector<cv::Point> right_edges_;
	// Prediction
	float offset_kalman_{0.0f};
	float angle_kalman_{0.0f};
	float estimated_lane_width_{-1.0f}; // when one lane edge is missing
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


	// cv::Mat measurement_;
	// cv::Mat prediction_;
	// Historical data
	int prev_left_edge_{320};
	int prev_right_edge_{320};
	int last_left_edge_pix_{320};
	int last_right_edge_pix_{320};

	// Store last known edge positions for smoothing
	// These will be used to maintain continuity in edge detection
	// This helps in cases where edges are not detected in every frame
	// and prevents sudden jumps in detected edge positions.
	// This is particularly useful in dynamic environments where lane edges may not be consistently visible.
	// The last known positions help to provide a reference point for the next frame's edge detection.
	// This is important for maintaining a smooth driving experience and avoiding abrupt steering corrections.
	// These values are updated only when valid edges are detected.
	// They are initialized to -1 to indicate that no edges have been detected yet.
	// If no edges are detected in the current frame, the last known positions will be used.
	// This helps to maintain a consistent lane detection experience.
	float last_left_edge_ = -1.0f;  // Store last known left edge position
	float last_right_edge_ = -1.0f; // Store last known right edge position
	// Lane geometry parameters
	// These parameters represent the geometry of the detected lane lines.
	// They are calculated based on the detected edges in the ROI.
	// The slopes and intercepts are used to define the lane lines in the image.
	// The slopes represent the angle of the lane lines with respect to the horizontal axis.
	// The intercepts represent the vertical position of the lane lines in the image.
	// These parameters are used to calculate the offset and angle of the lane with respect to the car's center.
	// The slopes and intercepts are calculated using linear regression on the detected edges.
	// These values are updated based on the detected edges in the ROI.
	// The slopes and intercepts are used to define the lane lines in the image.
	// The slopes and intercepts are used to calculate the offset and angle of the lane with respect to the car's center.
	// float left_slope_ = 0.0f;  // Slope of the left lane line
	// float right_slope_ = 0.0f; // Slope of the right lane line
	// float left_intercept_ = 0.0f;  // Intercept of the left lane line
	// float right_intercept_ = 0.0f; // Intercept of the right lane line

	// image vertical useful range of the edges
	float current_y_top_ = 0.0f; // Y-coordinate of the top of the ROI
	float current_y_bottom_ = 0.0f; // Y-coordinate of the bottom of the ROI
	float current_y_range_ = 0.0f;// = current_y_bottom_ - current_y_top_; // Range of Y-coordinates in the ROI

	//New members for lane geometry historical data
	std::vector<imgGeometry> history_; // History of lane geometry parameters
	static constexpr size_t MAX_HISTORY_SIZE = 10; // Maximum size of the history
	std::vector<float> lane_width_history_; // This is used to store the last known lane geometry width

	// Coefficients for distance calculation, converting pixels to meters
	// Equations :
	//  d(m) = s(y) * x
	// 	s(y) = a * y + b
	// x and y are pixel coordinates
	static constexpr double Asy = -2.6e-6; // Coefficient for distance calculation
	static constexpr double Bsy = 1.35e-3; // Coefficient for distance calculation



};

#endif // LANE_DETECTOR_HPP