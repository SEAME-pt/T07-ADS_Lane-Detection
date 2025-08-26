
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    int x1, y1, x2, y2;
    float conf;
    int class_id;
};

class ObjectDetector {
public:
    ObjectDetector(const std::string& json_file);
    ~ObjectDetector() = default;

    bool updateDetections();
    void drawDetections(cv::Mat& frame);

private:
    std::string json_file_;
    std::vector<Detection> detections_;
};
