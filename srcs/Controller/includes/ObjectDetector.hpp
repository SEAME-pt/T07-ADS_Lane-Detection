#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include "LaneDetector.hpp"

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};


class ObjectDetector {
public:
    ObjectDetector(const std::string& engine_path, int input_dim = 640);
    ~ObjectDetector();

    std::vector<Detection> infer(const cv::Mat& frame);

private:
    void preprocess(const cv::Mat& input, float* gpu_input, float& scale, int& dw, int& dh);
    std::vector<Detection> postprocess(const std::vector<float>& output, float scale, int dw, int dh,
                                       float conf_threshold, float nms_threshold);

    int inputDim;
    int numClasses;
    std::vector<std::string> classNames;

    Logger logger;
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    void* buffers[2]; // GPU buffers: input/output
    int inputIndex;
    int outputIndex;
    size_t inputSize;
    size_t outputSize;
    cudaStream_t stream;
};

#endif
