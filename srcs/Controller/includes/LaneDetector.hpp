#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

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
    void calculateLaneGeometry(float& offset, float& angle);

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
    cv::KalmanFilter kalman_;
    cv::Mat measurement_;
    cv::Mat prediction_;
    float offset_kalman_;
    float angle_kalman_;

    // Dimensões
    int input_width_ = 256;
    int input_height_ = 128;
    int frame_width_ = 640;
    int frame_height_ = 360;
    int roi_start_y_ = 40; // Top of ROI (bottom 320 pixels)
    int roi_end_y_ = 360;  // Bottom of frame
    float last_left_edge_;  // Store last known left edge position
    float last_right_edge_; // Store last known right edge position

    // Valores
    float estimated_lane_width_ = 200.0f;
};

#endif // LANE_DETECTOR_HPP