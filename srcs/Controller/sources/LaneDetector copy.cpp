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
    frame_height_ = 128;
    frame_width_ = 256;
    // frame_height_ = 360;
    // frame_width_ = 640;
    roi_start_y_ = 0;
    // roi_start_y_ = frame_height_ / 2;
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

// void LaneDetector::preprocess(const cv::Mat& frame) {
//     cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_);
//     cv::Mat cropped_frame = frame(roi);

//     float model_aspect = static_cast<float>(input_width_) / input_height_;
//     float crop_aspect = static_cast<float>(cropped_frame.cols) / cropped_frame.rows;

//     int resize_width, resize_height;
//     if (crop_aspect > model_aspect) {
//         resize_width = input_width_;
//         resize_height = static_cast<int>(input_width_ / crop_aspect);
//     } else {
//         resize_height = input_height_;
//         resize_width = static_cast<int>(input_height_ * crop_aspect);
//     }

//     gpu_frame_.upload(cropped_frame);
//     cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(resize_width, resize_height));
//     cv::Mat resized;
//     gpu_resized_.download(resized);

//     cv::Mat padded = cv::Mat::zeros(input_height_, input_width_, resized.type());
//     int pad_top = (input_height_ - resize_height) / 2;
//     int pad_left = (input_width_ - resize_width) / 2;
//     resized.copyTo(padded(cv::Rect(pad_left, pad_top, resize_width, resize_height)));

//     padded.convertTo(padded, CV_32F, 1.0 / 255.0);
//     std::vector<cv::Mat> channels;
//     cv::split(padded, channels);
//     for (int c = 0; c < 3; ++c) {
//         memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
//     }
// }

void LaneDetector::preprocess(const cv::Mat& frame) {
    // Converte BGR para RGB e normaliza para [0,1], redimensiona para a entrada do modelo.
    // blobFromImage faz tudo: resize, normalização, formatação CHW
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame,
        blob,
        1.0 / 255.0,
        cv::Size(input_width_, input_height_),
        cv::Scalar(), // sem subtração de média
        true,         // troca BGR -> RGB
        false         // não faz crop
    );

    // Copia o blob (1, 3, H, W) para o vetor input_data_ (assumindo float, CHW)
    std::memcpy(
        input_data_.data(),
        blob.ptr<float>(),
        input_width_ * input_height_ * 3 * sizeof(float)
    );
}

/// @brief		Run inference on the input data
/// @details	Uses the TensorRT context to perform inference on the input data.
/// @throws		std::runtime_error if the inference fails
/// @note		Checks that the context has been created and is ready for use.
/// @note		Checks that the input and output buffers have been allocated and are ready for use.
/// @note		Checks that the inference context has been successfully created.
/// @note		Copies the input data from the host to the GPU device.
/// @note		Copies the output data from the GPU device back to the host.
/// @note		Assumes that the input data has been preprocessed and is in the correct format.
/// @note		Assumes that the output data has been allocated and is ready to receive the results.
/// @note		Assumes that the CUDA stream has been created and is ready for use.
/// @note		Assumes that the input data is in the correct format and size for the model.
void LaneDetector::infer() {
	// Check if context and buffers are initialized
    if (!context_ || !buffers_[0] || !buffers_[1]) {
        throw std::runtime_error("Inference context or buffers not initialized.");
    }
	// copy the input data from the host to the GPU device
	cudaMemcpyAsync(
		buffers_[0],
		input_data_.data(),
		input_data_.size() * sizeof(float),
		cudaMemcpyHostToDevice,
		stream_
	);
	// run inference
    bool success = context_->enqueueV2(
								buffers_,
								stream_,
								nullptr
							);
	if (!success) {
		throw std::runtime_error("Inference failed!");
	}
	// copy the output data from GPU device back to the host
    cudaMemcpyAsync(
		output_data_.data(),
		buffers_[1],
		output_data_.size() * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream_
	);
	// Waits for it all to finish
    cudaStreamSynchronize(stream_);
}

/// @brief				Calculate the lane geometry based on the lane mask
/// @param	offset		Estimated offset of the lane center from the camera center
/// @param	angle		Estimated angle of the lane center with respect to the camera
/// @param	debug_img	Image for debugging purposes
/// @return true if lane geometry was successfully estimated, false otherwise
/// @return true if lane geometry was successfully estimated, false otherwise
/// @details
/// - Calculates the lane geometry by finding the left and right edges of the lane
/// - Uses a dense sampling approach to find the left and right edges of the lane
/// - Uses a weighted linear regression to estimate the lane geometry
/// - Uses a Kalman filter to smooth the estimated offset and angle
/// @note
/// - Uses a fixed :
/// -	 maximum distance to search for lane edges
/// -	 camera center based on the frame width
/// -	 camera tilt is 17 degrees and height is 15cm
/// -	 ROI height based on the input image height
/// -	 ROI width based on the input image width
/// -	 ROI start and end y-coordinates based on the input image height
/// -	 ROI aspect ratio based on the input image width and height
/// - Assumes that the lane mask is :
/// -	 a binary image with values in the range [0, 1]
/// -	 a single channel image
/// - Assumes that the Kalman filter is initialized with the correct :
/// -	 state transition matrix
/// - 	 measurement matrix
/// - 	 error covariance matrix
/// - 	 measurement noise covariance matrix
/// - 	 process noise covariance matrix
/// - 	 state vector
/// - 	 measurement vector
/// - 	 prediction vector
void LaneDetector::calculateLaneGeometry(float& offset, float& angle, cv::Mat& debug_img) {
    const int camera_center = frame_width_ / 2;
    const int max_distance = 350;
    std::vector<cv::Point2f> centers;
    std::vector<float> weights;

    cv::Mat mask_8u;
	    cv::Mat mask_8u;
    if (lane_mask_.type() != CV_8U) {
        lane_mask_.convertTo(mask_8u, CV_8U, 255.0);
    } else {
        mask_8u = lane_mask_;
    }
    // lane_mask_.convertTo(mask_8u, CV_8U, 255.0);//revisar esta linha
	// cv::Mat mask_8u = lane_mask_.clone();

    // Amostragem densa a cada 10 pixels da parte inferior da ROI
    for (int y = roi_end_y_ - 10; y >= roi_start_y_ + 30; y -= 10) {
        uchar* row = mask_8u.ptr<uchar>(y);
        int left = camera_center, right = camera_center;

        for (int x = camera_center; x >= std::max(0, camera_center - max_distance); --x) {
            if (row[x] > 0) {
                left = x;
                break;
            }
        }
        for (int x = camera_center; x < std::min(frame_width_, camera_center + max_distance); ++x) {
            if (row[x] > 0) {
                right = x;
                break;
            }
        }

        // if (left != camera_center && right != camera_center) {
        if (left < right) {
            float cx = (left + right) / 2.0f;
            centers.emplace_back(cx, y);

            // Peso proporcional à proximidade do fundo (mais baixo = mais peso)
            float w = static_cast<float>(y - roi_start_y_) / (roi_end_y_ - roi_start_y_);
            weights.push_back(w);
        }
    }

    // Se não houver pontos suficientes
    if (centers.size() < 3) {
        offset = offset_kalman_;
        angle = angle_kalman_;
        return;
    }

    // Ajuste de reta ponderada: x = a*y + b
    float sum_w = 0, sum_y = 0, sum_x = 0, sum_yx = 0, sum_yy = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        float w = weights[i];
        float x = centers[i].x;
        float y = centers[i].y;

        sum_w += w;
        sum_x += w * x;
        sum_y += w * y;
        sum_yx += w * y * x;
        sum_yy += w * y * y;
    }

    float denom = sum_w * sum_yy - sum_y * sum_y;
    float a = 0.0f, b = 0.0f;
    if (std::abs(denom) > 1e-5f) {
        a = (sum_w * sum_yx - sum_y * sum_x) / denom;
        b = (sum_x * sum_yy - sum_y * sum_yx) / denom;
    }

    // Estimar offset no ponto mais baixo da ROI
    float y_ref = roi_end_y_ - 10;
    float x_ref = a * y_ref + b;
    offset = x_ref - camera_center;

    // Estimar ângulo (em graus) da tangente: atan(dx/dy)
    angle = atan(a) * 180.0f / CV_PI;

    // Kalman Filter
    measurement_.at<float>(0) = offset;
    measurement_.at<float>(1) = angle;
    kalman_.correct(measurement_);
    prediction_ = kalman_.predict();
    offset_kalman_ = prediction_.at<float>(0);
    angle_kalman_ = prediction_.at<float>(2);

    offset = std::clamp(offset_kalman_, -frame_width_ / 2.0f, frame_width_ / 2.0f);
    angle = std::clamp(angle_kalman_, -90.0f, 90.0f);

    // Visualização
    if (!debug_img.empty()) {
        for (auto& pt : centers) {
            cv::circle(debug_img, pt, 3, cv::Scalar(0, 255, 255), -1);
        }

        cv::Point pt1(a * (y_ref - 100) + b, y_ref - 100);
        cv::Point pt2(a * y_ref + b, y_ref);
        cv::line(debug_img, pt1, pt2, cv::Scalar(255, 0, 255), 2);
    }
}

// Dentro de LaneDetector::processFrame
void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());

    // Sigmoid para converter logits em probabilidades
    cv::exp(-lane_mask_, lane_mask_);
    lane_mask_ = 1.0 / (1.0 + lane_mask_);

    // Resize e ajuste de ROI
    // int roi_height = roi_end_y_ - roi_start_y_;
    // float model_aspect = static_cast<float>(input_width_) / input_height_;
    // float roi_aspect = static_cast<float>(frame_width_) / roi_height;

    // int resize_width, resize_height;
    // if (roi_aspect > model_aspect) {
    //     resize_width = frame_width_;
    //     resize_height = static_cast<int>(frame_width_ / model_aspect);
    // } else {
    //     resize_height = roi_height;
    //     resize_width = static_cast<int>(roi_height * model_aspect);
    // }

    // cv::resize(lane_mask_, lane_mask_, cv::Size(resize_width, resize_height));

    // cv::Mat resized_mask;
    // if (resize_height > roi_height) {
    //     int crop_top = (resize_height - roi_height) / 2;
    //     resized_mask = lane_mask_(cv::Rect(0, crop_top, frame_width_, roi_height));
    // } else {
    //     resized_mask = cv::Mat::zeros(roi_height, frame_width_, lane_mask_.type());
    //     int pad_top = (roi_height - resize_height) / 2;
    //     lane_mask_.copyTo(resized_mask(cv::Rect(0, pad_top, frame_width_, resize_height)));
    // }

    // lane_mask_ = cv::Mat::zeros(frame_height_, frame_width_, lane_mask_.type());
    // resized_mask.copyTo(lane_mask_(cv::Rect(0, roi_start_y_, frame_width_, roi_height)));

    // Aplica threshold final
    lane_mask_ = (lane_mask_ > 0.3);

    // Resize da máscara para o tamanho do frame original
    cv::Mat resized_mask;
    cv::resize(lane_mask_, resized_mask, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_NEAREST);
    lane_mask_ = resized_mask;

    // Prepare output
    output_frame = frame.clone();

    // Calcular geometria da pista
    calculateLaneGeometry(offset, angle, output_frame);

    // Visualização do centro estimado da pista
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2;
    int lane_center = frame_width_ / 2 + static_cast<int>(offset);

    // Linha vermelha no centro da pista estimada
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2);

    // Linha azul no centro da imagem (referência)
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2);

    // Texto do offset e ângulo
    std::string offset_text = "Offset: " + std::to_string(static_cast<int>(offset));
    std::string angle_text = "Angle: " + std::to_string(static_cast<int>(angle)) + " deg";
    cv::putText(output_frame, offset_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(output_frame, angle_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Mostrar máscara pós-threshold no canto inferior direito
    if (visualize_mask) {
        cv::Mat threshold_mask_display;
        lane_mask_.convertTo(threshold_mask_display, CV_8U, 255.0);
        cv::resize(threshold_mask_display, threshold_mask_display, cv::Size(frame_width_ / 4, frame_height_ / 4));
        cv::cvtColor(threshold_mask_display, threshold_mask_display, cv::COLOR_GRAY2BGR);
        threshold_mask_display.copyTo(output_frame(cv::Rect(frame_width_ - threshold_mask_display.cols, frame_height_ - threshold_mask_display.rows, threshold_mask_display.cols, threshold_mask_display.rows)));
    }

    // Print no terminal
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[LaneDetector] Offset: " << offset
              << ", Angle: " << angle
              << " deg | Frame Size: " << frame.cols << "x" << frame.rows
              << std::endl;
}
