#ifndef OBJECTDETECTOR_HPP
#define OBJECTDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <map>
#include <vector>

// Estrutura para detecções
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

class ObjectDetector {
private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };
    
    std::vector<Buffer> inputBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<void*> bindings;
    
    int inputIndex;
    int outputIndex;
    int inputSize;
    int num_classes;
    size_t outputSize;
    cudaStream_t stream;
    void* buffers[2];
    
    std::map<int, std::string> classes;
    std::map<int, cv::Scalar> colors;
    float scale;  // Adicionado para compartilhar entre preprocess e postprocess
    int dw, dh;   // Adicionado para compartilhar entre preprocess e postprocess

    void allocateBuffers();

public:
    ObjectDetector(const std::string& engine_path, int input_sz = 640);
    ~ObjectDetector();
    
    void preprocess(const cv::Mat& input, float* gpu_input);
    std::vector<Detection> postprocess(float* gpu_output, const cv::Mat& frame);
    void processFrame(const cv::Mat& input, cv::Mat& output);
    
    cv::Scalar getColor(int class_id);
};

#endif // OBJECTDETECTOR_HPP