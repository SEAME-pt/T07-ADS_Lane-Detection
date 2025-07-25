#ifndef LANEDETECTOR_HPP
#define LANEDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <nvinfer1/nvinfer1.h>
#include "Debug.hpp"

#define ROI_SY_PERCENT 0.4f
#define ROI_EY_PERCENT 0.8f
#define ROI_X_BORDER 10
#define THRESHOLD 0.5f
#define MIN_EDGE_POINTS 10
#define MAX_HISTORY_SIZE 10
#define CAMERA_OFFSET 0.0f
#define Asy 0.001f
#define Bsy 0.1f
#define X_CAR_FRAME_CENTER 0.2f
#define X_CAR_FRAME_BOTTOM 0.5f

class LaneDetector {
public:
    LaneDetector(const std::string& trt_model_path);
    ~LaneDetector();
    bool initialize();
    void processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask = false);

private:
    struct imageGeometry {
        double left_slope, left_intercept;
        double right_slope, right_intercept;
        float lane_width;
        float offset, angle;
    } iGeo_;

    struct imageFrame {
        float xlbPX, xrbPX, xmbPX;
        float xltPX, xrtPX, xmtPX;
        float xcPX;
        float xmb, xmt;
    } imgFrame_;

    struct carFrame {
        float xT = X_CAR_FRAME_CENTER;
        float xB = X_CAR_FRAME_BOTTOM;
        float xDelta = X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM;
        float yT, yB;
        float slope, intercept;
    } carFrame_;

    std::vector<imageGeometry> history_;
    std::vector<float> lane_width_history_;
    float estimated_lane_width_;
    float offset_smooth_, angle_smooth_;
    float alpha_; // Low-pass filter coefficient

    int input_height_, input_width_;
    int frame_height_, frame_width_;
    int roi_sx_, roi_ex_, roi_sy_, roi_ey_;
    int roi_w_, roi_h_;
    float prev_left_edge_, prev_right_edge_;
    float last_left_edge_, last_right_edge_;
    cv::VideoCapture cap_;
    cv::Mat lane_mask_;
    std::vector<cv::Point> left_edges_, right_edges_;
    cudaStream_t stream_;
    std::vector<float> input_data_, output_data_;
    void* buffers_[2];
    std::unique_ptr<Debug> debug_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    nvinfer1::ILogger logger_;

    void defineROI();
    bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi);
    void weightedLinearRegression(const std::vector<cv::Point>& edges, double& slope, double& intercept);
    bool calculateMiddleLaneLine();
    void calculateOffsetAndAngle(float& offset, float& angle) const;
    bool calculateLaneGeometry(float& offset, float& angle);
    void loadEngine(const std::string& trt_model_path);
    void infer();
    void preprocess(const cv::Mat& frame);
};


#endif // LANEDETECTOR_HPP