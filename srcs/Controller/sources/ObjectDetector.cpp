#include "ObjectDetector.hpp"
#include <fstream>
#include <json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

ObjectDetector::ObjectDetector(const std::string& json_file) : json_file_(json_file) {}

bool ObjectDetector::updateDetections() {
    std::ifstream f(json_file_);
    if (!f.is_open()) return false;

    try {
        json j;
        f >> j;
        detections_.clear();
        for (auto& det : j) {
            Detection d;
            d.x1 = det["x1"];
            d.y1 = det["y1"];
            d.x2 = det["x2"];
            d.y2 = det["y2"];
            d.conf = det["conf"];
            d.class_id = det["class"];
            detections_.push_back(d);
        }
        return true;
    } catch (...) {
        return false;
    }
}

void ObjectDetector::drawDetections(cv::Mat& frame) {
    for (auto& d : detections_) {
        cv::rectangle(frame, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2),
                      cv::Scalar(0, 255, 0), 2);
    }
}
