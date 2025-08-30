#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include "LaneDetector.hpp"

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

using namespace nvinfer1;

// Estrutura para detecções
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

// Classe principal para YOLO TensorRT
class ObjectDetector {
private:
    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };

    std::vector<Buffer> inputBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<void*> bindings;

    int input_size;
    int num_classes;
    size_t output_size;

    std::map<int, std::string> classes;
    std::map<int, cv::Scalar> colors;

    void allocateBuffers();

public:
    ObjectDetector(const std::string& engine_path, int input_sz = 320);
    ~ObjectDetector();

    std::vector<float> preprocess(const cv::Mat& image, float& scale, int& dw, int& dh);
    std::vector<Detection> postprocess(const std::vector<float>& output, float scale, int dw, int dh,
                                       float conf_threshold = 0.3, float nms_threshold = 0.4);
    std::vector<Detection> infer(const cv::Mat& image, const cv::Rect& roi);

    cv::Scalar getColor(int class_id);
};

#endif // OBJECT_DETECTOR_HPP