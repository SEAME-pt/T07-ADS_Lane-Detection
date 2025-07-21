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

void Debug::showOutputVideo(cv::Mat& binary_mask, cv::Mat& output_frame, imgGeometry iGeo, int camera_offset) {
    // Ensure output_frame is valid
    if (output_frame.empty()) {
        output_frame = cv::Mat(frame_height_, frame_width_, CV_8UC3, cv::Scalar(0));
    }
    if (output_frame.type() != CV_8UC3) {
        output_frame.convertTo(output_frame, CV_8UC3);
    }

	// Draw angle and offset on top left corner
	std::string angle_text = "iGeo.yaw: " + std::to_string(iGeo.angle) + " deg";
	std::string offset_text = "iGeo.offset : " + std::to_string(iGeo.offset) + " m";
	cv::putText(output_frame, angle_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
	cv::putText(output_frame, offset_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    // Draw left and right lane lines using slopes and intercepts
    cv::Point lpt1(iGeo.left_slope * 0 + iGeo.left_intercept, 0 );
    // cv::Point lpt1(iGeo.left_slope * (frame_height_ / 2) + iGeo.left_intercept, (frame_height_ / 2) );
    cv::Point lpt2(iGeo.left_slope * (frame_height_ - 1)  + iGeo.left_intercept, (frame_height_ - 1));
    cv::line(output_frame, lpt1, lpt2, cv::Scalar(255, 0, 255), 2); // Purple for left lane

    cv::Point rpt1(iGeo.right_slope * 0 + iGeo.right_intercept, 0 );
    // cv::Point rpt1(iGeo.right_slope * (frame_height_ / 2) + iGeo.right_intercept, (frame_height_ / 2) );
    cv::Point rpt2(iGeo.right_slope * (frame_height_ - 1) + iGeo.right_intercept, (frame_height_ - 1));
    cv::line(output_frame, rpt1, rpt2, cv::Scalar(255, 0, 255), 2); // Purple for right lane

    // Calculate central lane points (ptm1 at roi_sy_, ptm2 at roi_ey_)
    float xl1 = iGeo.left_slope * (0)  + iGeo.left_intercept; // Left edge at top (roi_sy_)
    float xr1 = iGeo.right_slope * (0)  + iGeo.right_intercept; // Right edge at top (roi_sy_)
    float xm1 = (xr1 + xl1) / 2.0f; // Midpoint at top
    cv::Point ptm1(static_cast<int>(xm1), (0) );

    float xl2 = iGeo.left_slope * (frame_height_ - 1) + iGeo.left_intercept; // Left edge at bottom (roi_ey_)
    float xr2 = iGeo.right_slope * (frame_height_ - 1) + iGeo.right_intercept; // Right edge at bottom (roi_ey_)
    float xm2 = (xr2 + xl2) / 2.0f; // Midpoint at bottom
    cv::Point ptm2(static_cast<int>(xm2), (frame_height_ - 1) );

    // Draw central lane line (yellow, solid)
    cv::line(output_frame, ptm1, ptm2, cv::Scalar(0, 255, 255), 1); // Yellow for central lane
    // Draw central horizontal line (red, solid)
    cv::line(output_frame, cv::Point(0,frame_height_ / 2), cv::Point(frame_width_, frame_height_ / 2), cv::Scalar(255, 0, 0), 1); // Yellow for central lane

    // Draw camera center line (vertical, red)
	cv::Point xc1(camera_center_ - camera_offset, (0)) ; // Camera center adjusted by offset
	// cv::Point xc1(camera_center_ - camera_offset, (frame_height_ / 2)) ; // Camera center adjusted by offset
	cv::Point xc2(camera_center_ - camera_offset, (frame_height_ -1)); // Camera center adjusted by offset
    cv::line(output_frame, xc1, xc2, cv::Scalar(0, 0, 255), 1);

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

	// avoid the executionm of this function for debug purposes
	return;

	// normal behavior of this function
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