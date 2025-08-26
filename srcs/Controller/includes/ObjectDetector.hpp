#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, float confThreshold = 0.5);
    bool openCamera(int width = 640, int height = 480, int fps = 30);
    void runInferenceLoop();

private:
    std::string buildGStreamerPipeline(int width, int height, int fps);
    void drawPredictions(cv::Mat& frame, const cv::Mat& outs);

    cv::dnn::Net net;
    cv::VideoCapture cap;
    float confidenceThreshold;
    int inputWidth;
    int inputHeight;
};

#endif // OBJECT_DETECTOR_HPP
