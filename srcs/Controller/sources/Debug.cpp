#include "Debug.hpp"
#include <iomanip>
#include <fstream>

Debug::Debug(int frame_width, int frame_height, int roi_sy, int roi_ey)
    : frame_width_(frame_width),
      frame_height_(frame_height),
      roi_sy_(roi_sy),
      roi_ey_(roi_ey),
      camera_center_(frame_width / 2) {
}

Debug::~Debug() {
}

void Debug::showOutputVideo(cv::Mat& binary_mask, cv::Mat& output_frame, float left_slope, float left_intercept, float right_slope, float right_intercept, float angle, float offset, int camera_offset) {
    // Ensure output_frame is valid
    if (output_frame.empty()) {
        output_frame = cv::Mat(frame_height_, frame_width_, CV_8UC3, cv::Scalar(0));
    }
    if (output_frame.type() != CV_8UC3) {
        output_frame.convertTo(output_frame, CV_8UC3);
    }

	// Draw angle and offset on top left corner
	std::string camera_center_text = "CarCenter: " + std::to_string(camera_center_ + camera_offset) + " px";
	cv::putText(output_frame, camera_center_text, cv::Point(320, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	std::string angle_text = "Angle: " + std::to_string(angle * 180.0 / CV_PI) + " deg";
	std::string offset_text = "Offset: " + std::to_string(offset) + " m";
	cv::putText(output_frame, angle_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	cv::putText(output_frame, offset_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    // Draw left and right lane lines using slopes and intercepts
    cv::Point lpt1(left_slope * roi_sy_ + left_intercept, roi_sy_);
    cv::Point lpt2(left_slope * roi_ey_ + left_intercept, roi_ey_);
    cv::line(output_frame, lpt1, lpt2, cv::Scalar(255, 0, 255), 2); // Purple for left lane

    cv::Point rpt1(right_slope * roi_sy_ + right_intercept, roi_sy_);
    cv::Point rpt2(right_slope * roi_ey_ + right_intercept, roi_ey_);
    cv::line(output_frame, rpt1, rpt2, cv::Scalar(255, 0, 255), 2); // Purple for right lane

    // Calculate central lane points (ptm1 at roi_sy_, ptm2 at roi_ey_)
    float xl1 = left_slope * roi_sy_ + left_intercept; // Left edge at top (roi_sy_)
    float xr1 = right_slope * roi_sy_ + right_intercept; // Right edge at top (roi_sy_)
    float xm1 = (xr1 + xl1) / 2.0f; // Midpoint at top
    cv::Point ptm1(static_cast<int>(xm1), roi_sy_);

    float xl2 = left_slope * roi_ey_ + left_intercept; // Left edge at bottom (roi_ey_)
    float xr2 = right_slope * roi_ey_ + right_intercept; // Right edge at bottom (roi_ey_)
    float xm2 = (xr2 + xl2) / 2.0f; // Midpoint at bottom
    cv::Point ptm2(static_cast<int>(xm2), roi_ey_);

    // Draw central lane line (yellow, solid)
    cv::line(output_frame, ptm1, ptm2, cv::Scalar(0, 255, 255), 2); // Yellow for central lane

    // Draw camera center line (vertical, red)
    cv::line(output_frame, cv::Point(camera_center_ - camera_offset, roi_sy_), cv::Point(camera_center_ - camera_offset, roi_ey_), cv::Scalar(0, 0, 255), 2);

	// Draw mask thumb
	cv::Mat mask_thumb;
	cv::resize(binary_mask, mask_thumb, cv::Size(frame_width_ / 4, frame_height_ / 4), 0, 0, cv::INTER_NEAREST);
	cv::cvtColor(mask_thumb, mask_thumb, cv::COLOR_GRAY2BGR);
	mask_thumb.copyTo(output_frame(cv::Rect(frame_width_ - mask_thumb.cols, 0, mask_thumb.cols, mask_thumb.rows)));
}

void Debug::saveToFile(const std::string& filename,
                       const std::vector<cv::Point>& left_edges,
                       const std::vector<cv::Point>& right_edges,
                       float offset,
                       float angle,
                       const cv::Mat& lane_mask) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Offset: " << std::fixed << std::setprecision(2) << offset << " m\n";
    file << "Angle: " << std::fixed << std::setprecision(2) << (angle * 180.0 / CV_PI) << " deg\n";
    file << "Left Edges: ";
    for (const auto& pt : left_edges) {
        file << "(" << pt.x << ", " << pt.y << ") ";
    }
    file << "\nRight Edges: ";
    for (const auto& pt : right_edges) {
        file << "(" << pt.x << ", " << pt.y << ") ";
    }
    file << "\n----------------------------------------\n";
    file.close();
}