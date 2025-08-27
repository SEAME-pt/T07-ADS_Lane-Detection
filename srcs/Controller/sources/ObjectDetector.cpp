#include "ObjectDetector.hpp"
#include <fstream>
#include <cuda_runtime_api.h>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// Instância global usada pelo TensorRT
Logger gLogger;

ObjectDetector::ObjectDetector(const std::string& engine_path, int input_sz) : inputSize(input_sz) {
    // Carregar engine serializada
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "Erro ao abrir engine: " << engine_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ifstream::end);
    size_t engine_size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);
    file.close();

    // Criar runtime e engine
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);

    int nbBindings = engine->getNbBindings();
    std::cout << "[INFO] Número de bindings: " << nbBindings << std::endl;
    for (int i = 0; i < nbBindings; ++i) {
        std::string name = engine->getBindingName(i);
        bool isInput = engine->bindingIsInput(i);
        std::cout << "[Binding] Index " << i << ": "
                << name << " (" << (isInput ? "Input" : "Output") << ")"
                << std::endl;
    }

    context = engine->createExecutionContext();

    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");

    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);

    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i)
        inputSize *= inputDims.d[i];

    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i)
        outputSize *= outputDims.d[i];

    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));
    cudaStreamCreate(&stream);

    // Inicializações do código do amigo
    classes = {
        {0, "STOP"}, {1, "YIELD"}, {2, "SPEED_50"}, {3, "SPEED_80"},
        {4, "LIGHT_RED"}, {5, "LIGHT_GREEN"}, 
        {6, "LIGHT_YELLOW"}, {7, "CROSSWALK"}, {8, "DANGER"}, {9, "DANGER_CURVE"}
    };

    colors = {
        {0, cv::Scalar(255, 0, 0)}, {1, cv::Scalar(0, 255, 0)}, {2, cv::Scalar(0, 0, 255)}, 
        {3, cv::Scalar(255, 255, 0)}, {4, cv::Scalar(255, 0, 255)}, {5, cv::Scalar(0, 255, 255)}, 
        {6, cv::Scalar(128, 0, 128)}, {7, cv::Scalar(0, 128, 255)}, {8, cv::Scalar(128, 128, 0)}, 
        {9, cv::Scalar(255, 165, 0)}
    };

    num_classes = classes.size();
}

ObjectDetector::~ObjectDetector() {
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void ObjectDetector::allocateBuffers() {
    int nbBindings = engine->getNbBindings();
    inputBuffers.resize(1);
    outputBuffers.resize(1);
    bindings.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            vol *= dims.d[j];
        }
        size_t typeSize = sizeof(float);
        size_t totalSize = vol * typeSize;

        void* deviceMem;
        cudaMalloc(&deviceMem, totalSize);
        float* hostMem = new float[vol];

        bindings[i] = deviceMem;
        
        if (engine->bindingIsInput(i)) {
            inputBuffers[0] = {deviceMem, hostMem, totalSize};
        } else {
            outputBuffers[0] = {deviceMem, hostMem, totalSize};
            outputSize = vol;
        }
    }
}

void ObjectDetector::preprocess(const cv::Mat& input, float* gpu_input) {
    int h = input.rows;
    int w = input.cols;
    
    scale = std::min(static_cast<float>(inputSize) / w, static_cast<float>(inputSize) / h);
    int nw = static_cast<int>(scale * w);
    int nh = static_cast<int>(scale * h);
    
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(nw, nh));
    
    cv::Mat padded = cv::Mat::ones(inputSize, inputSize, CV_8UC3) * 114;
    dw = (inputSize - nw) / 2;
    dh = (inputSize - nh) / 2;
    
    resized.copyTo(padded(cv::Rect(dw, dh, nw, nh)));
    
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);
    
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    
    std::vector<float> inputData;
    for (int i = 0; i < 3; ++i) {
        inputData.insert(inputData.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
    }
    
    cudaMemcpyAsync(gpu_input, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

std::vector<Detection> ObjectDetector::postprocess(float* gpu_output, const cv::Mat& frame) {
    std::vector<float> output_host(outputSize);
    cudaMemcpyAsync(output_host.data(), gpu_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<Detection> detections;
    
    int total_elements = outputSize;
    int num_detections = total_elements / (4 + num_classes);
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    for (int i = 0; i < num_detections; ++i) {
        float cx = output_host[i];
        float cy = output_host[num_detections + i];
        float width = output_host[2 * num_detections + i];
        float height = output_host[3 * num_detections + i];
        
        float max_confidence = 0.0f;
        int best_class_id = 0;
        
        for (int j = 0; j < num_classes; ++j) {
            float class_conf = output_host[(4 + j) * num_detections + i];
            if (class_conf > max_confidence) {
                max_confidence = class_conf;
                best_class_id = j;
            }
        }
        
        if (max_confidence > 0.5f) {
            float x_center = (cx - dw) / scale;
            float y_center = (cy - dh) / scale;
            float w = width / scale;
            float h = height / scale;
            
            float x1 = x_center - w / 2.0f;
            float y1 = y_center - h / 2.0f;
            
            if (x1 >= 0 && y1 >= 0 && w > 0 && h > 0) {
                boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                                       static_cast<int>(w), static_cast<int>(h)));
                scores.push_back(max_confidence);
                class_ids.push_back(best_class_id);
            }
        }
    }
    
    if (!boxes.empty()) {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.4f, indices);
        
        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.class_name = classes.count(det.class_id) ? classes[det.class_id] : "unknown";
            det.confidence = scores[idx];
            det.bbox = boxes[idx];
            detections.push_back(det);
        }
    }
    
    return detections;
}

void ObjectDetector::processFrame(const cv::Mat& input, cv::Mat& output) {
    preprocess(input, static_cast<float*>(buffers[inputIndex]));
    context->enqueueV2(bindings.data(), stream, nullptr); // Usando bindings alocados
    std::vector<Detection> detections = postprocess(static_cast<float*>(buffers[outputIndex]), input);
    
    output = input.clone();
    
    for (const auto& det : detections) {
        cv::rectangle(output, det.bbox, getColor(det.class_id), 2);
        std::string label = det.class_name + " " + std::to_string(int(det.confidence * 100)) + "%";
        cv::putText(output, label, cv::Point(det.bbox.x, det.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        std::cout << "Classe " << det.class_id << ": " << 1 << " deteccoes\n";
    }
}

cv::Scalar ObjectDetector::getColor(int class_id) {
    return colors.count(class_id) ? colors[class_id] : cv::Scalar(255, 255, 255);
}