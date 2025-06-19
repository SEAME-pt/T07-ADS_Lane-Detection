#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>

// Logger para TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

// Função para carregar o modelo TensorRT
void loadEngine(const std::string& trt_model_path, nvinfer1::IRuntime*& runtime,
                nvinfer1::ICudaEngine*& engine, nvinfer1::IExecutionContext*& context,
                std::vector<void*>& buffers, std::vector<int>& binding_sizes,
                cudaStream_t& stream, int input_height, int input_width) {
    Logger logger;
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Error opening TensorRT model file: " + trt_model_path);
    }

    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");
    engine = runtime->deserializeCudaEngine(trt_model.data(), trt_model.size());
    if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
    context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("Failed to create TensorRT execution context");

    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) throw std::runtime_error("CUDA stream creation failed: " + std::string(cudaGetErrorString(err)));

    buffers.resize(engine->getNbBindings());
    binding_sizes.resize(engine->getNbBindings());

    // Aloca buffers CUDA
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        size *= sizeof(float);
        binding_sizes[i] = size;
        err = cudaMalloc(&buffers[i], size);
        if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for buffer " + std::to_string(i) + ": " + std::string(cudaGetErrorString(err)));
    }
}

// Função de pré-processamento (adaptada do seu preprocess)
void preprocess(const cv::Mat& frame, std::vector<float>& input_data, int input_width, int input_height) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_CUBIC);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB); // Converte BGR para RGB

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0); // Normaliza para [0,1]

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data.data() + c * input_height * input_width, channels[c].data, 
               input_height * input_width * sizeof(float));
    }

    // Depuração: Verificar valores RGB
    std::cout << "Input sample (R,G,B): " << input_data[0] << ", " 
              << input_data[input_height * input_width] << ", " 
              << input_data[2 * input_height * input_width] << std::endl;
}

// Função de inferência (adaptada do seu infer)
void infer(nvinfer1::IExecutionContext* context, std::vector<void*>& buffers, 
           cudaStream_t stream, std::vector<float>& input_data, 
           std::vector<float>& output_data) {
    cudaError_t err = cudaMemcpyAsync(buffers[0], input_data.data(), 
                                     input_data.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to device failed: " + std::string(cudaGetErrorString(err)));

    if (!context->enqueueV2(buffers.data(), stream, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    err = cudaMemcpyAsync(output_data.data(), buffers[1], 
                         output_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to host failed: " + std::string(cudaGetErrorString(err)));

    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA error after inference: " + std::string(cudaGetErrorString(err)));
}

// Função de pós-processamento (adaptada do seu processFrame)
void processFrame(const std::vector<float>& output_data, cv::Mat& lane_mask, 
                 cv::Mat& output_frame, const cv::Mat& frame, 
                 int input_height, int input_width, int frame_width, int frame_height) {
    lane_mask = cv::Mat(input_height, input_width, CV_32F, const_cast<float*>(output_data.data()));

    cv::Mat exp_mask;
    cv::exp(-lane_mask, exp_mask);
    lane_mask = 1.0 / (1.0 + exp_mask); // Aplica sigmoide

    double min_val, max_val;
    cv::minMaxLoc(lane_mask, &min_val, &max_val);
    std::cout << "lane_mask min: " << min_val << ", max: " << max_val << std::endl;

    cv::Mat binary_mask;
    float threshold = 0.5; // Limiar fixo
    cv::threshold(lane_mask, binary_mask, threshold, 1.0, cv::THRESH_BINARY);

    // Salva máscara antes da morfologia
    cv::Mat display_mask_pre_morph;
    binary_mask.convertTo(display_mask_pre_morph, CV_8U, 255);
    cv::imwrite("binary_mask_pre_morph.png", display_mask_pre_morph);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_DILATE, kernel);
    cv::morphologyEx(binary_mask, lane_mask, cv::MORPH_CLOSE, kernel);

    // Visualização lado a lado
    cv::Mat frame_resized;
    cv::resize(frame, frame_resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_CUBIC);

    cv::Mat display_mask;
    lane_mask.convertTo(display_mask, CV_8U, 255);
    cv::Mat display_mask_bgr;
    cv::cvtColor(display_mask, display_mask_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat side_by_side;
    cv::hconcat(frame_resized, display_mask_bgr, side_by_side);
    cv::imshow("Image and Mask", side_by_side);
    cv::waitKey(0); // Aguarda tecla para visualização estática

    // Redimensiona a máscara para o tamanho do frame
    cv::resize(lane_mask, lane_mask, cv::Size(frame_width, frame_height), 0, 0, cv::INTER_NEAREST);

    // Copia a imagem original para output_frame
    output_frame = frame.clone();

    // Salva saídas para depuração
    cv::imwrite("lane_mask.png", lane_mask * 255);
    cv::imwrite("binary_mask.png", binary_mask * 255);
    cv::imwrite("output_frame.png", output_frame);
    cv::imwrite("side_by_side.png", side_by_side);
}

int main() {
    // Dimensões
    const int input_height = 128;
    const int input_width = 256;
    const int frame_height = 360;
    const int frame_width = 640;

    // Buffers
    std::vector<float> input_data(3 * input_height * input_width);
    std::vector<float> output_data(input_height * input_width);
    std::vector<void*> buffers;
    std::vector<int> binding_sizes;
    cudaStream_t stream;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    try {
        // Carrega a imagem
        cv::Mat frame = cv::imread("test_image.jpg");
        if (frame.empty()) {
            throw std::runtime_error("Erro ao carregar a imagem!");
        }

        // Carrega o modelo TensorRT
        loadEngine("model.engine", runtime, engine, context, buffers, binding_sizes, stream, input_height, input_width);

        // Pré-processamento
        preprocess(frame, input_data, input_width, input_height);

        // Inferência
        infer(context, buffers, stream, input_data, output_data);

        // Pós-processamento
        cv::Mat lane_mask, output_frame;
        processFrame(output_data, lane_mask, output_frame, frame, input_height, input_width, frame_width, frame_height);

        // Limpeza
        for (void* buffer : buffers) {
            cudaFree(buffer);
        }
        cudaStreamDestroy(stream);
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;

        // Limpeza em caso de erro
        for (void* buffer : buffers) {
            cudaFree(buffer);
        }
        cudaStreamDestroy(stream);
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
        return -1;
    }

    return 0;
}