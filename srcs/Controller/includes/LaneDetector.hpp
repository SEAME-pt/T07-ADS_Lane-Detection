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

enum KalmanStateIndex { OFFSET = 0, OFFSET_VEL = 1, ANGLE = 2 };
enum KalmanMeasurementIndex { MEASUREMENT_OFFSET = 0, MEASUREMENT_ANGLE = 1 };
enum KalmanPredictionIndex { PREDICTION_OFFSET = 0, PREDICTION_ANGLE = 1 };
enum KalmanPredictionCovIndex { PREDICTION_COV_OFFSET = 0, PREDICTION_COV_ANGLE = 1 };
enum KalmanMeasurementCovIndex { MEASUREMENT_COV_OFFSET = 0, MEASUREMENT_COV_ANGLE = 1 };
enum KalmanErrorCovIndex { ERROR_COV_OFFSET = 0, ERROR_COV_ANGLE = 1 };
enum KalmanProcessCovIndex { PROCESS_COV_OFFSET = 0, PROCESS_COV_ANGLE = 1 };
enum KalmanTransitionIndex { TRANSITION_OFFSET = 0, TRANSITION_VEL = 1, TRANSITION_ANGLE = 2 };
enum KalmanMeasurementMatrixIndex { MEASUREMENT_MATRIX_OFFSET = 0, MEASUREMENT_MATRIX_ANGLE = 1 };

static constexpr int ROI_X_BORDER = 20; // Pixels from the left and right edges to avoid noise
static constexpr int I_W = 256;
static constexpr int I_H = 128;
static constexpr int F_W = 640; // Frame width
static constexpr int F_H = 360; // Frame height
static constexpr float ROI_SY_PERCENT = 0.5f; // ROI starts at 50% of image height
static constexpr float ROI_EY_PERCENT = 0.9f;   // ROI ends at 90% of image height
static constexpr double METER_PER_PIXEL = 0.0005556; // Example scale factor, should be calibrated [m/pixel]

// Coefficients for distance calculation, converting pixels to meters
// Equations :
//  d(m) = s(y) * x
// 	s(y) = a * y + b
// x and y are pixel coordinates
static constexpr double Asy = -4.57e-6; // Coefficient for distance calculation
static constexpr double Bsy = 1.98e-3;   // Coefficient for distance calculation


static constexpr double C_DISTANCE = 0.0001; // Coefficient for distance calculation
static constexpr double MIN_EDGE_POINTS = 10; // Coefficient for angle calculation
static constexpr double THRESHOLD = 0.3f; // Threshold for binary mask

static constexpr double X_IMG_ROI_TOP_CAR_FRAME = 0.40f; // X coordinate of the car center in the image frame
static constexpr double X_IMG_ROI_BOTTOM_CAR_FRAME = 0.f; // X coordinate of the bottom ROI in the image frame

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
    bool initialize();
    void processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask = false);
    cv::VideoCapture cap_;

private:
    void loadEngine(const std::string& trt_model_path);
    void preprocess(const cv::Mat& frame);
    void infer();
    // void calculateLaneGeometry(float& offset, float& angle);
    bool calculateLaneGeometry(float& offset, float& angle);
    // Helper functions
    void defineROI() ;
    // bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi,
    //                    std::vector<cv::Point>& left_edges,
    //                    std::vector<cv::Point>& right_edges) const;
    cv::Mat birdsEyeTransform(const cv::Mat& frame) const;
	bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) ;
    void weightedLinearRegression(const std::vector<cv::Point>& points,
                                  double& slope, double& intercept) ;
	double calculateThirdSegmentSlope(double x_start_left, double x_end_left,
                                 double x_start_right, double x_end_right,
                                 double x_start_3rd, double y_start, double y_end,
                                 double s_left_slope, double s_right_slope) const;
	// void calculateOffsetAndAngle(double left_slope, double left_intercept,
    //                              double right_slope, double right_intercept,
    //                              int y_bottom, float& offset, float& angle) const;
	void calculateOffsetAndAngle(float& offset, float& angle) const;
    void applyKalmanFilter(float measured_offset, float measured_angle,
                           float& smoothed_offset, float& smoothed_angle);

    // void drawDebugInfo(cv::Mat* debug_img, const std::vector<cv::Point>& left_edges,
    //                    const std::vector<cv::Point>& right_edges,
    //                    float offset, float angle) const;
	// double calculateDistance(int pixel_y, int x_length) const;

    // TensorRT
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Logger logger_;
    void* buffers_[2];
    cudaStream_t stream_;
    std::vector<float> input_data_;
    std::vector<float> output_data_;

    // OpenCV
    cv::Mat lane_mask_;

    // Kalman Filter
    cv::KalmanFilter kf_;         // Kalman filter for smoothing offset and angle
    // cv::KalmanFilter kalman_;
    cv::Mat measurement_;
    cv::Mat prediction_;
    float offset_kalman_;
    float angle_kalman_;
	std::vector<cv::Point> left_edges_;
	std::vector<cv::Point> right_edges_;
    // Valores
    float estimated_lane_width_;
    int prev_left_edge_;
    int prev_right_edge_;

    // Dimensions
    int input_width_ = I_W;
    int input_height_ = I_H;
    int frame_width_ = F_W;
    int frame_height_ = F_H;

	// Region of Interest (ROI) parameters
	// These parameters define the area of the image where lane detection will be performed.
	// The ROI is defined to focus on the bottom part of the image where lanes are typically located.
	// The ROI is set to start 20 pixels from the left edge and end 20 pixels from the right edge.
	// The vertical ROI starts at 40 pixels from the top and extends to the bottom of the frame.
	int roi_sx_, roi_ex_, roi_sy_, roi_ey_, roi_w_, roi_h_;


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
	float left_slope_ = 0.0f;  // Slope of the left lane line
	float right_slope_ = 0.0f; // Slope of the right lane line
	float left_intercept_ = 0.0f;  // Intercept of the left lane line
	float right_intercept_ = 0.0f; // Intercept of the right lane line
	// image vertical useful range of the edges
	float current_y_top_ = 0.0f; // Y-coordinate of the top of the ROI
	float current_y_bottom_ = 0.0f; // Y-coordinate of the bottom of the ROI
	float current_y_range_ = 0.0f;// = current_y_bottom_ - current_y_top_; // Range of Y-coordinates in the ROI

	// Fixed parameters as constants
    static constexpr double CAMERA_TILT = 19.0 * CV_PI / 180.0; // 19 degrees in radians (19 * pi/180)
	static constexpr double CAMERA_X_POS = 0.09f;    // 9 cm in meters, forward of the car's CM
	static constexpr double CAMERA_Y_POS = 0.0f;     // 0 cm in meters, centered on the car's CM
    static constexpr double CAMERA_Z_POS = 0.115;   // 11.5 cm in meters
	static constexpr double CAMERA_FOCAL_LENGTH = 0.00315; // Focal length in meters (2 mm)
	static constexpr int CAMERA_OFFSET = 5; // Offset in pixels, adjust if needed
    static constexpr double METER_PER_PIXEL = 0.0005556;  // Example scale factor, should be calibrated [m/pixel]

	// static constexpr double METER_PER_PIXEL = 0.00022224;  // Example scale factor, should be calibrated [m/pixel]
    static constexpr float ROI_START_Y_PERCENT = 0.5f; // ROI starts at 50% of image height
    static constexpr float ROI_END_Y_PERCENT = 0.9f;   // ROI ends at 80% of image height
    static constexpr int MAX_SEARCH_DISTANCE = 310;    // Max distance (pixels) to search for edges

    std::unique_ptr<Debug> debug_;
};

#endif // LANE_DETECTOR_HPP