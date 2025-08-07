#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Configs.hpp"
// #include "LaneDetector.hpp"


typedef struct s_imgGeometry {
	double left_slope{0.0};       // Slope of the left lane line
	double left_intercept{0.0};   // Intercept of the left lane line
	double right_slope{0.0};      // Slope of the right lane line
	double right_intercept{0.0};  // Intercept of the right lane line
	double middle_slope{0.0};      // Slope of the right lane line
	double middle_intercept{0.0};  // Intercept of the right lane line
	float offset{0.0f};           // Offset from the center of the lane
	float angle{0.0f};            // Angle of the lane in radians
	float lane_width{-1.0f};      // Width of the lane in meters
} imgGeometry;

class Debug {
public:
    // Constructor initializes with frame dimensions
    Debug(int frame_width, int frame_height, int roi_sy, int roi_ey);

    // Destructor
    ~Debug();

    // Display output video with lane information
	void showOutputVideo(cv::Mat& binary_mask, cv::Mat& output_frame, imgGeometry iGeo, int camera_offset = 0);
    // void showOutputVideo(cv::Mat& binary_mask, cv::Mat& output_frame, float left_slope, float left_intercept, float right_slope, float right_intercept, float angle, float offset, int camera_offset = 0);

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