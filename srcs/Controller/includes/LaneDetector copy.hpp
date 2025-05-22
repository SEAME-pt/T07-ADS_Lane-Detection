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
    void processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask = true);

    void loadEngine(const std::string& trt_model_path);
    void preprocess(const cv::Mat& frame);
    void infer();

    void calculateSteeringParams(int left_edge, int right_edge, int& lane_center, float& offset, float& angle);
    void calculateLaneGeometry(float& offset, float& angle, cv::Mat& debug_img);

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
    cv::VideoCapture cap_;
    cv::cuda::GpuMat gpu_frame_;
    cv::cuda::GpuMat gpu_resized_;
    cv::Mat lane_mask_;

    // Kalman Filter
    cv::KalmanFilter kalman_;
    cv::Mat measurement_;
    cv::Mat prediction_;
    float offset_kalman_;
    float angle_kalman_;

    // Dimensões
    int input_width_;
    int input_height_;
    int frame_width_ = 640;
    int frame_height_ = 360;
    int roi_start_y_;
    int roi_end_y_;

    // Valores
    float estimated_lane_width_;
    int prev_left_edge_;
    int prev_right_edge_;
};

#endif // LANE_DETECTOR_HPP
