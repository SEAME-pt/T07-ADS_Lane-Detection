#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
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
    void defineROI(int& start_y, int& end_y, int& start_x, int& end_x) const;
    bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi,
                       std::vector<cv::Point>& left_edges,
                       std::vector<cv::Point>& right_edges) const;
    void weightedLinearRegression(const std::vector<cv::Point>& points,
                                  double& slope, double& intercept) const;
    void calculateOffsetAndAngle(double left_slope, double left_intercept,
                                 double right_slope, double right_intercept,
                                 int y_bottom, float& offset, float& angle) const;
    void applyKalmanFilter(float measured_offset, float measured_angle,
                           float& smoothed_offset, float& smoothed_angle);
    void drawDebugInfo(cv::Mat* debug_img, const std::vector<cv::Point>& left_edges,
                       const std::vector<cv::Point>& right_edges,
                       float offset, float angle) const;
	double calculateDistance(int pixel_y, int x_length) const;

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

    // Valores
    float estimated_lane_width_;
    int prev_left_edge_;
    int prev_right_edge_;

    // Dimensões
    int input_width_ = 256;
    int input_height_ = 128;
    int frame_width_ = 640;
    int frame_height_ = 360;
    int roi_start_y_ = 40; // Top of ROI (bottom 320 pixels)
    int roi_end_y_ = 360;  // Bottom of frame
    float last_left_edge_;  // Store last known left edge position
    float last_right_edge_; // Store last known right edge position

        // Fixed parameters as constants
    static constexpr double CAMERA_TILT = 0.296706; // 17 degrees in radians (17 * pi/180)
    static constexpr double CAMERA_HEIGHT = 0.15;   // 15 cm in meters
    // static constexpr double METER_PER_PIXEL = 0.0005556;  // Example scale factor, should be calibrated [m/pixel]
    static constexpr double METER_PER_PIXEL = 0.00022224;  // Example scale factor, should be calibrated [m/pixel]
    static constexpr float ROI_START_Y_PERCENT = 0.7f; // ROI starts at 70% of image height
    static constexpr float ROI_END_Y_PERCENT = 1.0f;   // ROI ends at 100% of image height
    static constexpr int MAX_SEARCH_DISTANCE = 500;    // Max distance (pixels) to search for edges
	static constexpr double A_DISTANCE = -2.62e-6; // Coefficient for distance calculation
	static constexpr double B_DISTANCE = 1.4722e-3;   // Coefficient for distance calculation
};

#endif // LANE_DETECTOR_HPP