#include "ObjectDetector.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath> // para sigmoid

// Função sigmoide
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

ObjectDetector::ObjectDetector(const std::string& engine_path, int input_size)
    : inputDim(input_size), numClasses(10)
{
    classNames = {"STOP","YIELD","SPEED_50","SPEED_80","LIGHT_RED","LIGHT_GREEN",
                  "LIGHT_YELLOW","CROSSWALK","DANGER","DANGER_CURVE"};

    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) throw std::runtime_error("Cannot open engine file");

    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    context = engine->createExecutionContext();

    inputIndex = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");

    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);

    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) inputSize *= std::max(1, inputDims.d[i]);

    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= std::max(1, outputDims.d[i]);

    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

    std::cout << "[INFO] Engine loaded. InputSize=" << inputSize << " OutputSize=" << outputSize << std::endl;
}

ObjectDetector::~ObjectDetector()
{
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

void ObjectDetector::preprocess(const cv::Mat& input, float* gpu_input, float& scale, int& dw, int& dh)
{
    int h = input.rows;
    int w = input.cols;

    scale = std::min(static_cast<float>(inputDim)/w, static_cast<float>(inputDim)/h);
    int nw = static_cast<int>(w * scale);
    int nh = static_cast<int>(h * scale);

    dw = (inputDim - nw)/2;
    dh = (inputDim - nh)/2;

    cv::Mat resized, padded(inputDim, inputDim, CV_8UC3, cv::Scalar(114,114,114));
    cv::resize(input, resized, cv::Size(nw, nh));
    resized.copyTo(padded(cv::Rect(dw, dh, nw, nh)));

    padded.convertTo(padded, CV_32FC3, 1.0/255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);

    size_t channel_size = inputDim * inputDim;
    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(gpu_input + i*channel_size, channels[i].ptr<float>(), channel_size*sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "[DEBUG] Preprocess done: scale=" << scale << ", dw=" << dw << ", dh=" << dh << std::endl;
}

std::vector<Detection> ObjectDetector::postprocess(const std::vector<float>& output, float scale, int dw, int dh,
                                                   float conf_threshold, float nms_threshold)
{
    std::vector<Detection> detections;

    int num_detections = output.size() / (4 + numClasses);
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    std::cout << "[DEBUG] Raw detections: " << num_detections << std::endl;

    for (int i = 0; i < num_detections; ++i) {
        // Extraindo os valores brutos do modelo
        float tx = output[i];
        float ty = output[num_detections + i];
        float tw = output[2*num_detections + i];
        float th = output[3*num_detections + i];

        // Aplicando sigmoid no centro
        float cx = (sigmoid(tx) * inputDim - dw) / scale;
        float cy = (sigmoid(ty) * inputDim - dh) / scale;

        // Aplicando sigmoid na largura e altura (simplificado para YOLOv11 sem anchors)
        float w = sigmoid(tw) * inputDim / scale;
        float h = sigmoid(th) * inputDim / scale;

        // Encontrar classe com maior confiança
        float max_conf = 0.0f;
        int best_class = -1;
        for (int j = 0; j < numClasses; ++j) {
            float class_conf = sigmoid(output[(4+j)*num_detections + i]);
            if (class_conf > max_conf) {
                max_conf = class_conf;
                best_class = j;
            }
        }

        std::cout << "[DEBUG] Detection " << i
                  << " | class=" << best_class
                  << " conf=" << max_conf
                  << " cx=" << cx
                  << " cy=" << cy
                  << " w=" << w
                  << " h=" << h << std::endl;

        if (max_conf > conf_threshold) {
            float x1 = std::max(0.0f, cx - w/2.0f);
            float y1 = std::max(0.0f, cy - h/2.0f);

            boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                     static_cast<int>(w), static_cast<int>(h)));
            confidences.push_back(max_conf);
            class_ids.push_back(best_class);
        }
    }

    // NMS para filtrar caixas sobrepostas
    if (!boxes.empty()) {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.class_name = classNames[det.class_id];
            det.confidence = confidences[idx];
            det.bbox = boxes[idx];
            detections.push_back(det);

            std::cout << "[INFO] Final detection: class=" << det.class_id
                      << " conf=" << det.confidence
                      << " box=(" << det.bbox.x << "," << det.bbox.y
                      << "," << det.bbox.width << "," << det.bbox.height << ")" << std::endl;
        }
    } else {
        std::cout << "[INFO] No detections above threshold" << std::endl;
    }

    return detections;
}



std::vector<Detection> ObjectDetector::infer(const cv::Mat& frame)
{
    float scale;
    int dw, dh;
    float* gpu_input = (float*)buffers[inputIndex];

    preprocess(frame, gpu_input, scale, dw, dh);

    // Zerar o buffer de saída antes de inferir
    cudaMemset(buffers[outputIndex], 0, outputSize * sizeof(float));

    bool success = context->executeV2(buffers);
    if (!success) {
        std::cerr << "[ERROR] TensorRT executeV2 failed!" << std::endl;
        return {};
    }

    float* gpu_output = (float*)buffers[outputIndex];
    std::vector<float> output(outputSize);

    // Copiar output da GPU para CPU
    cudaMemcpy(output.data(), gpu_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[DEBUG] Inference executed, output copied. Output size: " << output.size() << std::endl;

    // Imprimir alguns valores brutos para verificar
    std::cout << "[DEBUG] Raw output sample: ";
    for (int i = 0; i < std::min(10, (int)output.size()); ++i)
        std::cout << output[i] << " ";
    std::cout << std::endl;

    // Aplicar sigmoid para objectness + class confidence se YOLOv11
    for (int i = 0; i < outputSize; ++i) {
        output[i] = 1.0f / (1.0f + expf(-output[i]));
    }

    return postprocess(output, scale, dw, dh, 0.5f, 0.4f);
}

