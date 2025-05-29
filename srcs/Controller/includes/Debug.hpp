#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Debug {
public:
    // Constructor initializes with frame dimensions
    Debug(int frame_width, int frame_height, int roi_sy, int roi_ey);

    // Destructor
    ~Debug();

    // Display output video with lane information
    void showOutputVideo(cv::Mat& output_frame, 
                        const std::vector<cv::Point>& left_edges, 
                        const std::vector<cv::Point>& right_edges, 
                        float offset, 
                        float angle,
                        const cv::Mat& lane_mask,
                        bool visualize_mask);

    // Save debug information to a file
    void saveToFile(const std::string& filename, 
                    const std::vector<cv::Point>& left_edges, 
                    const std::vector<cv::Point>& right_edges, 
                    float offset, 
                    float angle,
                    const cv::Mat& lane_mask);

private:
    int frame_width_;
    int frame_height_;
    int roi_sy_;
    int roi_ey_;
    int camera_center_;
};

#endif // DEBUG_HPP