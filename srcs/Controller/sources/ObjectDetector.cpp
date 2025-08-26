#include "ObjectDetector.hpp"
#include <iostream>

ObjectDetector::ObjectDetector(const std::string& modelPath, float confThreshold)
    : confidenceThreshold(confThreshold), inputWidth(640), inputHeight(640) {
    try {
        net = cv::dnn::readNet(modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const std::exception& e) {
        std::cerr << "Erro ao carregar modelo: " << e.what() << std::endl;
        exit(1);
    }
}

std::string ObjectDetector::buildGStreamerPipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
           ", height=" + std::to_string(height) + ", format=NV12, framerate=" +
           std::to_string(fps) + "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
           "videoconvert ! video/x-raw, format=BGR ! appsink drop=true";
}

bool ObjectDetector::openCamera(int width, int height, int fps) {
    std::string pipeline = buildGStreamerPipeline(width, height, fps);
    cap.open(pipeline, cv::CAP_GSTREAMER);
    return cap.isOpened();
}

void ObjectDetector::drawPredictions(cv::Mat& frame, const cv::Mat& detections) {
    for (int i = 0; i < detections.rows; ++i) {
        float confidence = detections.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detections.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detections.at<float>(i, 4) * frame.rows);
            int xRightTop   = static_cast<int>(detections.at<float>(i, 5) * frame.cols);
            int yRightTop   = static_cast<int>(detections.at<float>(i, 6) * frame.rows);
            cv::rectangle(frame, cv::Point(xLeftBottom, yLeftBottom),
                          cv::Point(xRightTop, yRightTop),
                          cv::Scalar(0, 255, 0), 2);
        }
    }
}

void ObjectDetector::runInferenceLoop() {
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera CSI." << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Falha ao capturar frame." << std::endl;
            break;
        }

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                              cv::Size(inputWidth, inputHeight),
                                              cv::Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        for (auto& out : outputs) {
            drawPredictions(frame, out);
        }

        cv::imshow("YOLOv8 - Câmera CSI", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}
