#include "LaneDetector.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);

    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 1, 0, 0,
                                                         0, 1, 0, 0,
                                                         0, 0, 1, 1,
                                                         0, 0, 0, 1);
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(0.03));
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0));
    measurement_ = cv::Mat(2, 1, CV_32F);
    prediction_ = cv::Mat(4, 1, CV_32F);

    input_height_ = 128;
    input_width_ = 256;
    frame_height_ = 360;
    frame_width_ = 640;
    roi_start_y_ = frame_height_ / 2;
    roi_end_y_ = frame_height_ - 10;

    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = 200.0f;
    prev_left_edge_ = frame_width_ / 2;
    prev_right_edge_ = frame_width_ / 2;

    loadEngine(trt_model_path);
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool LaneDetector::initialize() {
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);
    return cap_.isOpened();
}

void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening TensorRT model file!" << std::endl;
        return;
    }

    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr));
    context_.reset(engine_->createExecutionContext());

    cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
    cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

void LaneDetector::preprocess(const cv::Mat& frame) {
    cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_);
    cv::Mat cropped_frame = frame(roi);

    float model_aspect = static_cast<float>(input_width_) / input_height_;
    float crop_aspect = static_cast<float>(cropped_frame.cols) / cropped_frame.rows;

    int resize_width, resize_height;
    if (crop_aspect > model_aspect) {
        resize_width = input_width_;
        resize_height = static_cast<int>(input_width_ / crop_aspect);
    } else {
        resize_height = input_height_;
        resize_width = static_cast<int>(input_height_ * crop_aspect);
    }

    gpu_frame_.upload(cropped_frame);
    cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(resize_width, resize_height));
    cv::Mat resized;
    gpu_resized_.download(resized);

    cv::Mat padded = cv::Mat::zeros(input_height_, input_width_, resized.type());
    int pad_top = (input_height_ - resize_height) / 2;
    int pad_left = (input_width_ - resize_width) / 2;
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, resize_width, resize_height)));

    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

void LaneDetector::infer() {
    cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->enqueueV2(buffers_, stream_, nullptr);
    cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void LaneDetector::findLaneEdges(int& left_edge, int& right_edge) {
    cv::Mat mask_8u;
    lane_mask_.convertTo(mask_8u, CV_8U, 255.0);
    int center_x = frame_width_ / 2;
    left_edge = center_x;
    right_edge = center_x;

    const int max_distance = 350;
    std::vector<int> left_edges, right_edges;

    for (int y = roi_start_y_; y < roi_end_y_; ++y) {
        uchar* row = mask_8u.ptr<uchar>(y);
        int temp_left = center_x;
        int temp_right = center_x;

        for (int x = center_x; x >= std::max(0, center_x - max_distance); --x) {
            if (row[x] > 0) {
                temp_left = x;
                break;
            }
        }
        for (int x = center_x; x < std::min(frame_width_, center_x + max_distance); ++x) {
            if (row[x] > 0) {
                temp_right = x;
                break;
            }
        }

        if (temp_left != center_x && abs(temp_left - prev_left_edge_) < 100) {
            left_edges.push_back(temp_left);
        }
        if (temp_right != center_x && abs(temp_right - prev_right_edge_) < 100) {
            right_edges.push_back(temp_right);
        }
    }

    if (left_edges.empty() && right_edges.empty()) {
        left_edge = prev_left_edge_;
        right_edge = prev_right_edge_;
    } else {
        if (!left_edges.empty()) {
            left_edge = std::accumulate(left_edges.begin(), left_edges.end(), 0) / left_edges.size();
            prev_left_edge_ = left_edge;
        } else {
            left_edge = prev_left_edge_;
        }
        if (!right_edges.empty()) {
            right_edge = std::accumulate(right_edges.begin(), right_edges.end(), 0) / right_edges.size();
            prev_right_edge_ = right_edge;
        } else {
            right_edge = prev_right_edge_;
        }
    }
}

void LaneDetector::calculateDualOffsets(int left_edge, int right_edge,
                                        int& lane_center_top, float& offset_top, float& angle_top,
                                        int& lane_center_bottom, float& offset_bottom, float& angle_bottom) {
    const int camera_center = frame_width_ / 2;
    int roi_top_y = (roi_start_y_ + roi_end_y_) / 2;
    int roi_bottom_y = roi_end_y_ - 10;

    int lane_width = right_edge - left_edge;
    if (lane_width < 30 || lane_width > 500) lane_width = estimated_lane_width_;

    estimated_lane_width_ = 0.9f * estimated_lane_width_ + 0.1f * lane_width;

    lane_center_top = (left_edge + right_edge) / 2;
    offset_top = static_cast<float>(lane_center_top - camera_center);
    angle_top = atan2(offset_top, frame_height_ - roi_top_y) * 180.0 / CV_PI;

    lane_center_bottom = (left_edge + right_edge) / 2;
    offset_bottom = static_cast<float>(lane_center_bottom - camera_center);
    angle_bottom = atan2(offset_bottom, frame_height_ - roi_bottom_y) * 180.0 / CV_PI;
}

void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::exp(-lane_mask_, lane_mask_);
    lane_mask_ = 1.0 / (1.0 + lane_mask_);

    int roi_height = roi_end_y_ - roi_start_y_;
    float model_aspect = static_cast<float>(input_width_) / input_height_;
    float roi_aspect = static_cast<float>(frame_width_) / roi_height;

    int resize_width, resize_height;
    if (roi_aspect > model_aspect) {
        resize_width = frame_width_;
        resize_height = static_cast<int>(frame_width_ / model_aspect);
    } else {
        resize_height = roi_height;
        resize_width = static_cast<int>(roi_height * model_aspect);
    }

    cv::resize(lane_mask_, lane_mask_, cv::Size(resize_width, resize_height));

    cv::Mat resized_mask;
    if (resize_height > roi_height) {
        int crop_top = (resize_height - roi_height) / 2;
        resized_mask = lane_mask_(cv::Rect(0, crop_top, frame_width_, roi_height));
    } else {
        resized_mask = cv::Mat::zeros(roi_height, frame_width_, lane_mask_.type());
        int pad_top = (roi_height - resize_height) / 2;
        lane_mask_.copyTo(resized_mask(cv::Rect(0, pad_top, frame_width_, resize_height)));
    }

    lane_mask_ = cv::Mat::zeros(frame_height_, frame_width_, lane_mask_.type());
    resized_mask.copyTo(lane_mask_(cv::Rect(0, roi_start_y_, frame_width_, roi_height)));
    lane_mask_ = (lane_mask_ > 0.3);

    int left_edge, right_edge;
    findLaneEdges(left_edge, right_edge);
    int lane_center_top, lane_center_bottom;
    float offset_top, angle_top, offset_bottom, angle_bottom;

    calculateDualOffsets(left_edge, right_edge,
                         lane_center_top, offset_top, angle_top,
                         lane_center_bottom, offset_bottom, angle_bottom);

    measurement_.at<float>(0) = offset_bottom;
    measurement_.at<float>(1) = angle_bottom;
    kalman_.correct(measurement_);
    prediction_ = kalman_.predict();
    offset_kalman_ = prediction_.at<float>(0);
    angle_kalman_ = prediction_.at<float>(2);

    offset = std::clamp(offset_kalman_, -frame_width_ / 2.0f, frame_width_ / 2.0f);
    angle = std::clamp(angle_kalman_, -90.0f, 90.0f);

    output_frame = frame.clone();
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2;
    cv::line(output_frame, cv::Point(left_edge, roi_mid_y), cv::Point(left_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    cv::line(output_frame, cv::Point(right_edge, roi_mid_y), cv::Point(right_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    int lane_center = (left_edge + right_edge) / 2;
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2);
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2);

    char text[128];
    sprintf(text, "Offset: %.2f px", offset);
    cv::putText(output_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    sprintf(text, "Angle: %.2f deg", angle);
    cv::putText(output_frame, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    cv::line(output_frame, cv::Point(0, roi_start_y_), cv::Point(frame_width_, roi_start_y_), cv::Scalar(255, 255, 0), 1);
    cv::line(output_frame, cv::Point(0, roi_end_y_), cv::Point(frame_width_, roi_end_y_), cv::Scalar(255, 255, 0), 1);

    if (visualize_mask) {
        cv::Mat mask_display;
        lane_mask_.convertTo(mask_display, CV_8U, 255.0);
        cv::resize(mask_display, mask_display, cv::Size(frame_width_ / 4, frame_height_ / 4));
        cv::cvtColor(mask_display, mask_display, cv::COLOR_GRAY2BGR);
        mask_display.copyTo(output_frame(cv::Rect(frame_width_ - mask_display.cols, frame_height_ - mask_display.rows, mask_display.cols, mask_display.rows)));
    }
}
