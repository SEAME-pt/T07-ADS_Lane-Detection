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

void Debug::showOutputVideo(cv::Mat& output_frame, 
                           const std::vector<cv::Point>& left_edges, 
                           const std::vector<cv::Point>& right_edges, 
                           float offset, 
                           float angle,
                           const cv::Mat& lane_mask,
                           bool visualize_mask) {
    // Ensure output_frame is valid
    if (output_frame.empty()) {
        output_frame = cv::Mat(frame_height_, frame_width_, CV_8UC3, cv::Scalar(0));
    }
    if (output_frame.type() != CV_8UC3) {
        output_frame.convertTo(output_frame, CV_8UC3);
    }

    // Convert lane mask to visualization format
    cv::Mat mask_vis;
    lane_mask.convertTo(mask_vis, CV_8U, 255.0);
    cv::Mat mask_color = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
    int max_distance = 300;

    // Find lane edges at sampled y-positions
    std::vector<int> left_edges_y, right_edges_y, valid_y;
    for (int y = roi_ey_ - 10; y >= roi_sy_; y -= 10) {
        uchar* row = mask_vis.ptr<uchar>(y);
        int left = camera_center_, right = camera_center_;

        for (int x = camera_center_; x >= std::max(0, camera_center_ - max_distance); --x) {
            if (row[x] > 128) {
                left = x;
                break;
            }
        }
        for (int x = camera_center_; x < std::min(frame_width_, camera_center_ + max_distance); ++x) {
            if (row[x] > 128) {
                right = x;
                break;
            }
        }
        if (left != camera_center_ || right != camera_center_) {
            left_edges_y.push_back(left != camera_center_ ? left : -1);
            right_edges_y.push_back(right != camera_center_ ? right : -1);
            valid_y.push_back(y);
        }
    }

    // Handle cases where edges are not detected
    static int last_valid_left = camera_center_;
    static int last_valid_right = camera_center_;
    static float avg_lane_width = 200.0f;

    if (!left_edges_y.empty() && !right_edges_y.empty()) {
        for (size_t i = 0; i < left_edges_y.size(); ++i) {
            if (left_edges_y[i] != -1) last_valid_left = left_edges_y[i];
            if (right_edges_y[i] != -1) last_valid_right = right_edges_y[i];
            if (left_edges_y[i] != -1 && right_edges_y[i] != -1 && right_edges_y[i] - left_edges_y[i] > 50) {
                avg_lane_width = 0.9 * avg_lane_width + 0.1 * (right_edges_y[i] - left_edges_y[i]);
            }
        }
    }

    if (left_edges_y.empty() || right_edges_y.empty()) {
        float x_ref = camera_center_ + offset;
        float lane_angle_rad = angle;
        for (int y = roi_ey_ - 10; y >= roi_sy_; y -= 10) {
            float dy = (y - (roi_ey_ - 10));
            float dx = dy * std::tan(lane_angle_rad);
            int estimated_center = static_cast<int>(x_ref + dx);
            left_edges_y.push_back(std::max(0, estimated_center - static_cast<int>(avg_lane_width / 2)));
            right_edges_y.push_back(std::min(frame_width_ - 1, estimated_center + static_cast<int>(avg_lane_width / 2)));
            valid_y.push_back(y);
        }
    } else {
        for (size_t i = 0; i < left_edges_y.size(); ++i) {
            if (left_edges_y[i] == -1) {
                if (i > 0 && left_edges_y[i-1] != -1) {
                    float slope = (last_valid_left - left_edges_y[i-1]) / (valid_y[i-1] - valid_y[i]);
                    left_edges_y[i] = static_cast<int>(left_edges_y[i-1] + slope * (valid_y[i-1] - valid_y[i]));
                } else {
                    float x_ref = camera_center_ + offset;
                    float lane_angle_rad = angle;
                    float dy = (valid_y[i] - (roi_ey_ - 10));
                    float dx = dy * std::tan(lane_angle_rad);
                    int estimated_center = static_cast<int>(x_ref + dx);
                    left_edges_y[i] = std::max(0, estimated_center - static_cast<int>(avg_lane_width / 2));
                }
            } else {
                last_valid_left = left_edges_y[i];
            }
            if (right_edges_y[i] == -1) {
                if (i > 0 && right_edges_y[i-1] != -1) {
                    float slope = (last_valid_right - right_edges_y[i-1]) / (valid_y[i-1] - valid_y[i]);
                    right_edges_y[i] = static_cast<int>(right_edges_y[i-1] + slope * (valid_y[i-1] - valid_y[i]));
                } else {
                    float x_ref = camera_center_ + offset;
                    float lane_angle_rad = angle;
                    float dy = (valid_y[i] - (roi_ey_ - 10));
                    float dx = dy * std::tan(lane_angle_rad);
                    int estimated_center = static_cast<int>(x_ref + dx);
                    right_edges_y[i] = std::min(frame_width_ - 1, estimated_center + static_cast<int>(avg_lane_width / 2));
                }
            } else {
                last_valid_right = right_edges_y[i];
            }
            if (right_edges_y[i] - left_edges_y[i] < 50) {
                right_edges_y[i] = left_edges_y[i] + 50;
            }
        }
    }

    // Draw lane mask
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y = valid_y[i];
        int left = left_edges_y[i];
        int right = right_edges_y[i];
        if (right - left < 50) right = left + 50;
        for (int x = left; x <= right && x < frame_width_; ++x) {
            mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
    }

    // Interpolate between y positions
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y_start = valid_y[i];
        int y_end = (i + 1 < valid_y.size()) ? valid_y[i + 1] : y_start;
        int left_start = left_edges_y[i];
        int right_start = right_edges_y[i];
        int left_end = (i + 1 < valid_y.size()) ? left_edges_y[i + 1] : left_start;
        int right_end = (i + 1 < valid_y.size()) ? right_edges_y[i + 1] : right_start;

        for (int y = y_start; y <= y_end; ++y) {
            float t = (y_end == y_start) ? 0.0f : static_cast<float>(y - y_start) / (y_end - y_start);
            int left = left_start + static_cast<int>(t * (left_end - left_start));
            int right = right_start + static_cast<int>(t * (right_end - right_start));
            if (right - left < 50) right = left + 50;
            for (int x = left; x <= right && x < frame_width_; ++x) {
                mask_color.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
            }
        }
    }

    // Overlay mask on output frame
    cv::addWeighted(output_frame, 0.7, mask_color, 0.5, 0.0, output_frame);

    // Draw lane centers and fit line
    std::vector<cv::Point2f> centers;
    std::vector<float> weights;
    for (size_t i = 0; i < valid_y.size(); ++i) {
        int y = valid_y[i];
        int left = left_edges_y[i];
        int right = right_edges_y[i];
        if (right - left > 50) {
            float cx = (left + right) / 2.0f;
            centers.emplace_back(cx, y);
            float w = static_cast<float>(y - roi_sy_) / (roi_ey_ - roi_sy_);
            weights.push_back(w * w);
        }
    }
    for (const auto& pt : centers) {
        cv::circle(output_frame, pt, 4, cv::Scalar(0, 255, 255), -1);
    }
    if (centers.size() >= 3) {
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
        if (std::abs(denom) > 1e-5f) {
            float a = (sum_w * sum_yx - sum_y * sum_x) / denom;
            float b = (sum_x * sum_yy - sum_y * sum_yx) / denom;
            int y_top = roi_ey_ - 100;
            cv::Point pt1(a * y_top + b, y_top);
            cv::Point pt2(a * (roi_ey_ - 10) + b, roi_ey_ - 10);
            cv::line(output_frame, pt1, pt2, cv::Scalar(255, 0, 255), 2);
        }
    }

    // Draw lane center line
    int lane_center = frame_width_ / 2 + static_cast<int>(offset);
    int line_y = (roi_sy_ + roi_ey_) / 2;
    cv::line(output_frame, cv::Point(lane_center, line_y), cv::Point(lane_center, line_y - 50), cv::Scalar(0, 0, 255), 3);

    // Draw offset line
    int frame_center = frame_width_ / 2;
    int distance = std::abs(static_cast<int>(offset));
    cv::line(output_frame, cv::Point(frame_center, frame_height_ - 1), cv::Point(lane_center, frame_height_ - 2), cv::Scalar(0, 255, 255), 2);
    std::string distance_text = "Distance: " + std::to_string(distance) + " px";
    cv::putText(output_frame, distance_text, cv::Point(frame_center + offset / 2 - 50, frame_height_ - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw lane width at bottom
    int left_edge_bottom = left_edges_y[0];
    int right_edge_bottom = right_edges_y[0];
    if (right_edge_bottom - left_edge_bottom < 50) right_edge_bottom = left_edge_bottom + 50;
    cv::line(output_frame, cv::Point(left_edge_bottom, frame_height_ - 1), cv::Point(right_edge_bottom, frame_height_ - 1), 
             cv::Scalar(255, 255, 0), 2);
    int lane_width_bottom = right_edge_bottom - left_edge_bottom;
    std::string width_text_bottom = "Lane Width (Bottom): " + std::to_string(lane_width_bottom) + " px";
    cv::putText(output_frame, width_text_bottom, cv::Point(left_edge_bottom + (right_edge_bottom - left_edge_bottom) / 2 - 50, frame_height_ - 25), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw lane width at center
    int center_y = (roi_sy_ + roi_ey_) / 2;
    int left_edge_center = camera_center_;
    int right_edge_center = camera_center_;
    float t = 0.0f;
    size_t center_index = 0;

    for (size_t i = 0; i < valid_y.size(); ++i) {
        if (valid_y[i] <= center_y) {
            center_index = i;
            break;
        }
    }
    if (center_index > 0 && center_index < valid_y.size()) {
        float y1 = valid_y[center_index - 1];
        float y2 = valid_y[center_index];
        t = (center_y - y1) / (y2 - y1);
        left_edge_center = static_cast<int>(left_edges_y[center_index - 1] + t * (left_edges_y[center_index] - left_edges_y[center_index - 1]));
        right_edge_center = static_cast<int>(right_edges_y[center_index - 1] + t * (right_edges_y[center_index] - right_edges_y[center_index - 1]));
    } else if (center_index == 0 && !valid_y.empty()) {
        left_edge_center = left_edges_y[0];
        right_edge_center = right_edges_y[0];
    } else if (center_index == valid_y.size() && !valid_y.empty()) {
        left_edge_center = left_edges_y.back();
        right_edge_center = right_edges_y.back();
    }

    if (right_edge_center - left_edge_center < 50) right_edge_center = left_edge_center + 50;
    cv::line(output_frame, cv::Point(left_edge_center, center_y), cv::Point(right_edge_center, center_y), cv::Scalar(255, 255, 0), 2);
    int lane_width_center = right_edge_center - left_edge_center;
    std::string width_text_center = "Lane Width (Center): " + std::to_string(lane_width_center) + " px";
    cv::putText(output_frame, width_text_center, cv::Point(left_edge_center + (right_edge_center - left_edge_center) / 2 - 50, center_y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw offset and angle text
    std::string offset_text = "Offset: " + std::to_string(static_cast<int>(offset)) + " px";
    cv::putText(output_frame, offset_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::string angle_text = "Angle: " + std::to_string(static_cast<int>(angle * 180.0 / CV_PI)) + " deg";
    cv::putText(output_frame, angle_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    // Visualize mask thumbnail if requested
    if (visualize_mask) {
        cv::Mat mask_thumb;
        cv::resize(mask_vis, mask_thumb, cv::Size(frame_width_ / 4, frame_height_ / 4), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(mask_thumb, mask_thumb, cv::COLOR_GRAY2BGR);
        mask_thumb.copyTo(output_frame(cv::Rect(frame_width_ - mask_thumb.cols, 0, mask_thumb.cols, mask_thumb.rows)));
    }
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

    file << "Offset: " << std::fixed << std::setprecision(2) << offset << " px\n";
    file << "Angle: " << std::fixed << std::setprecision(2) << (angle * 180.0 / CV_PI) << " deg\n";
    file << "Left Edges: ";
    for (const auto& pt : left_edges) {
        file << "(" << pt.x << ", " << pt.y << ") ";
    }
    file << "\nRight Edges: ";
    for (const auto& pt : right_edges) {
        file << "(" << pt.x << ", " << pt.y << ") ";
    }
    file << "\nLane Mask Size: " << lane_mask.cols << "x" << lane_mask.rows << "\n";
    file << "----------------------------------------\n";
    file.close();
}