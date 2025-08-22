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
#include <deque>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "Configs.hpp"

// struct imgGeometry {
//     float xlt{0.0f};
//     float xrt{0.0f};
//     float xlb{0.0f};
//     float xrb{0.0f};
//     float xmt{0.0f};
//     float xmb{0.0f};
//     float slope{0.0f};
//     float intercept{0.0f};
// };

struct LaneCandidate {
    std::vector<cv::Point> left_points;
    std::vector<cv::Point> right_points;
    double slope_left = 0.0;
    double slope_right = 0.0;
    double intercept_left = 0.0;
    double intercept_right = 0.0;
    double score = 0.0;
};

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
    void defineROI();
    bool calculateLaneGeometry(float& offset, float& angle, bool visualize_mask);
    bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi);
    void weightedLinearRegression(const std::vector<cv::Point>& points, double& slope, double& intercept);
    void calculateMiddleLaneLine();
    void calculateOffsetAndAngle(float& offset, float& angle);
    void applyKalmanFilter(float measured_offset, float measured_angle, float& smoothed_offset, float& smoothed_angle);
    void applyLowPassFilter(float measured_offset, float measured_angle, float& smoothed_offset, float& smoothed_angle);
    float smoothValue(std::deque<float>& history, float new_value);
    void detectLaneCandidates(const cv::Mat& binary_mask, std::vector<LaneCandidate>& candidates);
    LaneCandidate selectBestLane(const std::vector<LaneCandidate>& candidates);
    bool validateLane(const LaneCandidate& candidate);
    void updateHistory(const LaneCandidate& selected_lane);

    // Kalman Filter
    cv::KalmanFilter kf_;
    float offset_kalman_{0.0f};
    float angle_kalman_{0.0f};
    cv::Mat measurement_;
    cv::Mat prediction_;

    // Lane history
    std::vector<imgGeometry> history_;
    std::vector<float> lane_width_history_;
    float estimated_lane_width_;
    std::deque<float> missing_left_top_history_;
    std::deque<float> missing_left_bottom_history_;
    std::deque<float> missing_right_top_history_;
    std::deque<float> missing_right_bottom_history_;
    std::deque<LaneCandidate> lane_history_;
    LaneCandidate previous_lane_;

    // Filters
    float offset_smooth_;
    float angle_smooth_;
    float alpha_;

    // Dimensions
    int input_height_{I_H};
    int input_width_{I_W};
    int frame_height_{F_H};
    int frame_width_{F_W};

    // ROI
    int roi_sx_{ROI_X_BORDER};
    int roi_sy_{static_cast<int>(F_H * ROI_SY_PERCENT)};
    int roi_ex_{F_W - ROI_X_BORDER};
    int roi_ey_{static_cast<int>(F_H * ROI_EY_PERCENT)};
    int roi_w_{F_W - 2 * ROI_X_BORDER};
    int roi_h_{roi_ey_ - roi_sy_};

    // Historical data
    int prev_left_edge_;
    int prev_right_edge_;
    float last_left_edge_;
    float last_right_edge_;
    cv::Mat lane_mask_;
    std::vector<cv::Point> left_edges_, right_edges_;

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
    imgGeometry iGeo_;
    t_carFrame carFrame_;
    t_imgFrame imgFrame_;

    // Image vertical useful range
    float current_y_top_;
    float current_y_bottom_;
    float current_y_range_;
};

#endif // LANE_DETECTOR_HPP