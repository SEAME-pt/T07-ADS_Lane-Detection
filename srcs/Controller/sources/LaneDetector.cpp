#include "LaneDetector.hpp"


LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);

    // Initialize Kalman filter
    kf_ = cv::KalmanFilter(2, 2, 0, CV_32F);
    kf_.statePre.at<float>(0) = 0.0f;
    kf_.statePre.at<float>(1) = 0.0f;
    kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPre, cv::Scalar::all(1));

    input_height_ = 128;
    input_width_ = 256;
    frame_height_ = 360; // Corrected to match input frame
    frame_width_ = 640;  // Corrected to match input frame
    roi_sy_ = static_cast<int>(frame_height_ * ROI_START_Y_PERCENT); // 252
    roi_ey_ = static_cast<int>(frame_height_ * ROI_END_Y_PERCENT);     // 360

    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = 200.0f;

    prev_left_edge_ = frame_width_ / 2;  // 320
    prev_right_edge_ = frame_width_ / 2; // 320
    last_left_edge_ = frame_width_ / 2;  // 320
    last_right_edge_ = frame_width_ / 2; // 320

	defineROI();
    debug_ = std::make_unique<Debug>(frame_width_, frame_height_, roi_sy_, roi_ey_);

    loadEngine(trt_model_path);
	std::cout << "LaneDetector created with model: " << trt_model_path << std::endl;
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool LaneDetector::initialize() {

	// std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
    //                        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    //                        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
	std::string pipeline = "nvarguscamerasrc exposuretimerange=\"1000000 50000000\" gainrange=\"1 16\" !"
	                       "video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);
	if (!cap_.isOpened()) {
		std::cerr << "Failed to open camera pipeline!" << std::endl;
		return false;
	}
	std::cout << "[" << __func__ << "] "
				<< "Camera pipeline opened successfully: \n"
				<< pipeline << std::endl;
	std::cout << "[" << __func__ << "] "
				<< "LaneDetector initialization concluded!"
				<< std::endl;
    return cap_.isOpened();
}

bool LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
    // Check if lane mask is valid
    if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
        return false;
    }

    // Step 1: Define the Region of Interest (ROI)
    cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);
	if (roi.width <= 0 || roi.height <= 0) {
		std::cerr << "Invalid ROI dimensions!" << std::endl;
		return false; // Invalid ROI dimensions
	}
	// std::cout << "[" << __func__ << "] : ROI : \n"
	// 		  << "roi.x = " << roi.x << std::endl
	// 		  << "roi.y = " << roi.y << std::endl
	// 		  << "roi.width = " << roi.width << std::endl
	// 		  << "roi.height = " << roi.height << std::endl;


    // Step 2: Find left and right lane edges using dense sampling
    // std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi)) {
		std::cerr << "Not enough edge points detected in ROI!" << std::endl;
        return false; // Not enough edge points detected
    }

    // Step 3: Perform weighted linear regression to fit lines to edges
    // double left_slope, left_intercept, right_slope, right_intercept;
    weightedLinearRegression(left_edges_, iGeo_.left_slope, iGeo_.left_intercept);
	// iGeo_.left_slope = left_slope;
	// iGeo_.left_intercept = left_intercept; // Reset right intercept to zero
    weightedLinearRegression(right_edges_, iGeo_.right_slope, iGeo_.right_intercept);
	// right_slope_ = right_slope;
	// right_intercept_ = right_intercept; // Reset right intercept to zero
	// std::cout << "[" << __func__ << "] "
	// 			<< "Left : slope = " << iGeo_.left_slope << " | intercept = " << iGeo_.left_intercept
	// 			<< " || "
	// 			<< "Right : slope = " << iGeo_.right_slope << " | intercept = " << iGeo_.right_intercept << std::endl;



    // Step 4: Calculate offset and angle from the fitted lines
    float measured_offset, measured_angle;
    // calculateOffsetAndAngle(left_slope, left_intercept, right_slope, right_intercept,
    //                         roi_ey_, measured_offset, measured_angle);
	calculateOffsetAndAngle(measured_offset, measured_angle);
	// std::cout << "[" << __func__ << "] "
	// 		  << "\n\tMeasured Offset: " << std::setw(6) << measured_offset << " m,"
	// 		  << "\n\tMeasured Angle: " << std::setw(6) << measured_angle << " rad"
	// 		  << '\r' << std::flush;


	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;
    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;

	iGeo_.angle = angle * 180 / CV_PI; // Store angle in imgGeometry
	iGeo_.offset = offset ; // Store offset in imgGeometry

	//std::cout << "Offset: " << offset << " m, Angle: " << angle << " rad" << std::endl;

	// std::cout << "[" << __func__ << "] "
	// 		  << "Offset: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_offset
	// 		  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << offset << ") m"
	// 		  << "Angle: M(" << std::fixed << std::setprecision(4) << std::setw(6) << measured_angle
	// 		  << ") K(" << std::fixed << std::setprecision(4) << std::setw(6) << smoothed_angle << ") rad"
	// 		  << '\r' << std::flush;


    return true;
}

void LaneDetector::defineROI() {
		std::cout << "Defining ROI..." << std::endl;
		roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT); // 252 for 360
		roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);     // 360
		roi_sx_ = ROI_X_BORDER; // 20
		roi_ex_ = frame_width_ - ROI_X_BORDER; // 620
		roi_w_ = roi_ex_ - roi_sx_; // 620 - 20 = 600
		roi_h_ = roi_ey_ - roi_sy_; // 360 - 252 = 108
		std::cout << "ROI: "
					<< "sy = " << roi_sy_
					<< ", ey = " << roi_ey_
					<< ", sx = " << roi_sx_
					<< ", ex = " << roi_ex_
					<< ", w = " << roi_w_
					<< ", h = " << roi_h_
					<< std::endl;
}

cv::Mat LaneDetector::birdsEyeTransform(const cv::Mat& rawLane) const {

	cv::Mat warped_frame = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
	if (rawLane.empty()) {
		std::cerr << "Input frame is empty!" << std::endl;
		return warped_frame; // Return an empty matrix if the input is empty
	}
	std::cout << "[" << __func__ << "] : "
			  << "Raw lane size: " << rawLane.size()
			  << ", Type: " << rawLane.type()
			  << ", Channels: " << rawLane.channels() << std::endl;

	cv::Mat lane_mask_8u ;
	rawLane.convertTo(lane_mask_8u, CV_8U, 255.0);

	cv::Mat lane_mask_color;
	cv:cvtColor(lane_mask_8u, lane_mask_color, cv::COLOR_GRAY2BGR);
	// Define sources point based on rawLane size
	int rawLaneWidth = lane_mask_color.cols;
	int rawLaneHeight = lane_mask_color.rows;
	if (rawLaneWidth <= 0 || rawLaneHeight <= 0) {
		std::cerr << "Invalid lane_mask_color dimensions!" << std::endl;
		return warped_frame; // Return an empty matrix if the input is invalid
	}
	// Define the source points for the perspective transform
	std::vector<cv::Point2f> src_points = {
		cv::Point2f(100,
					static_cast<int>(rawLaneHeight * ROI_START_Y_PERCENT)), // Top-left
		cv::Point2f(rawLaneWidth - (100),
					static_cast<int>(rawLaneHeight * ROI_START_Y_PERCENT)), // Top-right
		cv::Point2f(rawLaneWidth - ROI_X_BORDER,
					static_cast<int>(rawLaneHeight * ROI_END_Y_PERCENT)), // Bottom-right
		cv::Point2f(ROI_X_BORDER,
					static_cast<int>(rawLaneHeight * ROI_END_Y_PERCENT))  // Bottom-left
	};

	// Define the destination points for the perspective transform
	std::vector<cv::Point2f> dst_points = {
		cv::Point2f(0, 0),             // Top-left
		cv::Point2f(frame_width_, 0),  // Top-right
		cv::Point2f(frame_width_, frame_height_), // Bottom-right
		cv::Point2f(0, frame_height_)   // Bottom-left
	};

	// Compute the perspective transform matrix
	cv::Mat T = cv::getPerspectiveTransform(src_points, dst_points);

	// Apply the perspective warp to the input frame
	cv::Mat birdEyeMask;
	cv::warpPerspective(lane_mask_color, birdEyeMask, T, lane_mask_color.size());

	// cv::Mat lane_mask_8u;
	// lane_mask_.convertTo(lane_mask_8u, CV_8U, 255.0);
    // cv::imwrite("lane_mask.png", lane_mask_8U * 255);


	cv::imwrite("birdEyeMask1.png", birdEyeMask);
	// Normalize the warped frame to the range [0, 1]
	cv::Mat normalized_mask;
	cv::normalize(birdEyeMask, normalized_mask, 0, 1, cv::NORM_MINMAX, CV_32F);
	// Convert the normalized mask to a single channel float image
	cv::Mat float_mask;
	cv::cvtColor(normalized_mask, float_mask, cv::COLOR_BGR2GRAY);
	// Convert the single channel float image to a 3-channel float image
	cv::Mat float_mask_3ch;
	cv::cvtColor(float_mask, float_mask_3ch, cv::COLOR_GRAY2BGR);
	// Resize the warped frame to the desired output size
	cv::resize(float_mask_3ch, warped_frame, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_LINEAR);
	// Convert the warped frame to a 32-bit float image
	if (warped_frame.type() != CV_32F) {
		std::cout << "[" << __func__ << "] : "
				  << "Converting warped frame to CV_32F type." << std::endl;
		warped_frame.convertTo(warped_frame, CV_32F);
		if (warped_frame.empty()) {
			std::cerr << "Failed to convert warped frame to CV_32F!" << std::endl;
			return cv::Mat(); // Return an empty matrix if the conversion fails
		}
	}
	// Normalize the warped frame to the range [0, 1]
	cv::normalize(warped_frame, warped_frame, 0, 1, cv::NORM_MINMAX, CV_32F);
	// Debugging output
	std::cout << "[" << __func__ << "] : "
			  << "Warped frame size: " << warped_frame.size()
			  << ", Type: " << warped_frame.type()
			  << ", Channels: " << warped_frame.channels() << std::endl;
	// Return the warped frame
	if (warped_frame.empty()) {
		std::cerr << "Warped frame is empty!" << std::endl;
		return cv::Mat(); // Return an empty matrix if the warped frame is empty
	} else if (warped_frame.type() != CV_32F) {
		std::cerr << "Warped frame type is not CV_32F!" << std::endl;
		return cv::Mat(); // Return an empty matrix if the warped frame type is not CV_32F
	} else if (warped_frame.channels() != 3) {
		std::cerr << "Warped frame does not have 3 channels!" << std::endl;
		return cv::Mat(); // Return an empty matrix if the warped frame does not have 3 channels
	} else if (warped_frame.size() != cv::Size(frame_width_, frame_height_)) {
		std::cerr << "Warped frame size does not match expected size!" << std::endl;
		return cv::Mat(); // Return an empty matrix if the warped frame size does not match expected size
	}
	else {
		std::cout << "[" << __func__ << "] : "
			  << "Warped frame successfully created with size: " << warped_frame.size()
			  << ", Type: " << warped_frame.type()
			  << ", Channels: " << warped_frame.channels() << std::endl;
	}
	// Return the warped frame
	return warped_frame;
}

bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi) {
    left_edges_.clear();
    right_edges_.clear();

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        int left_x = -1, right_x = -1;
        for (int x = roi.width / 2; x >= roi.x + ROI_X_BORDER; --x) {
            if (lane_mask.at<float>(y, x) > THRESHOLD) {
                left_x = x;
				left_edges_.emplace_back(left_x, y);
                break;
            }
        }
        for (int x = roi.width / 2; x <= roi.width - ROI_X_BORDER; ++x) {
            if (lane_mask.at<float>(y, x) > THRESHOLD) {
                right_x = x;
				right_edges_.emplace_back(right_x, y);
                break;
            }
        }
        // if (left_x != -1 && right_x != -1 && left_x < right_x) {
        //     left_edges_.emplace_back(left_x, y);
        //     right_edges_.emplace_back(right_x, y);
        // }
    }

	// Check if we found enough edges
	if (left_edges_.size() < MIN_EDGE_POINTS || right_edges_.size() < MIN_EDGE_POINTS) {
		std::cerr << "[" << __func__ << "] : "
				  << "Not enough edge points found in ROI!" << std::endl;
		return false; // Not enough edge points found
	}
	// else {
	// 	std::cout << "[" << __func__ << "] : "
	// 			  << "Found " << left_edges_.size() << " left edges and "
	// 			  << right_edges_.size() << " right edges in the ROI." << std::endl;
	// 	std::cout << "[" << __func__ << "] : "
	// 			  << "Left edges: " << left_edges_.size()
	// 			  << ", Right edges: " << right_edges_.size() << std::endl;
	// 	std::cout << "[" << __func__ << "] : "
	// 			  << "Left edge first: (" << left_edges_[0].x << ", " << left_edges_[0].y << ")"
	// 			  << ", Right edge first: (" << right_edges_[0].x << ", " << right_edges_[0].y << ")"
	// 			  << std::endl;
	// 	std::cout << "[" << __func__ << "] : "
	// 			  << "Left edge last: (" << left_edges_.back().x << ", " << left_edges_.back().y << ")"
	// 			  << ", Right edge last: (" << right_edges_.back().x << ", " << right_edges_.back().y << ")"
	// 			  << std::endl;
	// }

    return !left_edges_.empty() && !right_edges_.empty();
}

void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges,
                                            double& slope, double& intercept) {
    if (edges.size() < MIN_EDGE_POINTS) {
        slope = 0.0;
        intercept = frame_width_ / 2.0 - CAMERA_OFFSET; // 320
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int y_top = static_cast<int>(edges[0].y); // 252
    int y_bottom =static_cast<int>(current_y_top_ + edges.size());     // 360
    int y_range = y_bottom - y_top;

    for (const auto& pt : edges) {
        double y = pt.y;
        double x = pt.x;
        double weight = (y - y_top) / y_range;
        weight = std::max(0.1, weight);

        sum_w += weight;
        sum_wy += weight * y;
        sum_wx += weight * x;
        sum_wyy += weight * y * y;
        sum_wyx += weight * y * x;
    }

    double mean_y = sum_wy / sum_w;
    double mean_x = sum_wx / sum_w;
    double denom = sum_wyy - 2 * mean_y * sum_wy + sum_w * mean_y * mean_y;
    if (std::abs(denom) < 1e-6) {
        slope = 0.0;
        intercept = mean_x;
    } else {
        slope = (sum_wyx - mean_y * sum_wx - mean_x * sum_wy + sum_w * mean_x * mean_y) / denom;
        intercept = mean_x - slope * mean_y;
    }
}

double LaneDetector::calculateThirdSegmentSlope(double xLeftTop, double xLeftBottom,
                                 double xRightTop, double xRightBottom,
                                 double xCarTop, double yRoiTop, double yRoiBottom,
                                 double s_left_slope, double s_right_slope) const {
    // Validate xCarTop is between xLeftTop and xRightTop
    if (xCarTop < std::min(xLeftTop, xRightTop) || xCarTop > std::max(xLeftTop, xRightTop)) {
        std::cout << "[" << __func__ << "] : Error: xCarTop (" << xCarTop
                  << ") is outside range [" << xLeftTop << ", " << xRightTop << "]" << std::endl;
        // Clamp xCarTop to the nearest boundary
        xCarTop = std::max(xLeftTop, std::min(xRightTop, xCarTop));
    }

    // Calculate the relative position of xCarTop
    double delta_x_start = xRightTop - xLeftTop;
    if (std::abs(delta_x_start) < 1e-6) {
        std::cout << "[" << __func__ << "] : Error: xLeftTop and xRightTop are too close" << std::endl;
        return (xLeftBottom + xRightBottom) / 2.0 - xCarTop / (yRoiBottom - yRoiTop); // Fallback slope
    }
    double t = (xCarTop - xLeftTop) / delta_x_start;

    // Interpolate xCarBottom between xLeftBottom and xRightBottom
    double xCarBottom = xLeftBottom + t * (xRightBottom - xLeftBottom);

    // Validate xCarBottom is between xLeftBottom and xRightBottom
    if (xCarBottom < std::min(xLeftBottom, xRightBottom) || xCarBottom > std::max(xLeftBottom, xRightBottom)) {
        std::cout << "[" << __func__ << "] : Warning: xCarBottom (" << xCarBottom
                  << ") is outside range [" << xLeftBottom << ", " << xRightBottom << "]" << std::endl;
        xCarBottom = std::max(xLeftBottom, std::min(xRightBottom, xCarBottom));
    }

    // Calculate the slope of the third segment
    double yDelta = yRoiBottom - yRoiTop;
    if (std::abs(yDelta) < 1e-6) {
        std::cout << "[" << __func__ << "] : Error: yRoiBottom and yRoiTop are too close" << std::endl;
        return 0.0; // Avoid division by zero
    }

    double sCarSlope = (xCarBottom - xCarTop) / yDelta;
	//double isCarSlope = yDelta / (xCarBottom - xCarTop); // Inverse slope for debugging



    return sCarSlope;
}

// void LaneDetector::calculateOffsetAndAngle(double left_slope, double left_intercept,
//                                            double right_slope, double right_intercept,
//                                            int y_bottom, float& offset, float& angle) const {
void LaneDetector::calculateOffsetAndAngle(float& offset, float& angle) const {
	// All values in pixels
    int xc = frame_width_ / 2 + CAMERA_OFFSET; // Camera center adjusted by offset
    // Calculate edge points at bottom and top (adjusted by ROI_START_Y_PERCENT)
    int xlb = iGeo_.left_slope * frame_height_ + iGeo_.left_intercept ;//edges_[current_y_range_].x;
    int xrb = iGeo_.right_slope * frame_height_ + iGeo_.right_intercept;//edges_[current_y_range_].x;
    int xmb = xc - (xlb + xrb) / 2; // Midpoint distance to car center at bottom

	// int xlb = left_edges_[current_y_range_].x;
    int xlt = iGeo_.left_slope * (frame_height_ / 2) + iGeo_.left_intercept; //edges_[0].x;
    int xrt = iGeo_.right_slope * (frame_height_ / 2) + iGeo_.right_intercept;
	int xmt = xc - (xlt + xrt) / 2; // Midpoint distance to car center at top

	// Convert image midlane points [pixels] to image Frame midlane [meters]
	// using the equation d = s(y[pixels]) * x_img(pixel)
	// Points at center of image
	double xmt_imgFrame = (Asy * roi_sy_ + Bsy) * xmt;
	//Point at bottom of image
	double xmb_imgFrame = (Asy * frame_height_ + Bsy) * xmb;

	// Debugging output
	std::cout << "[" << __func__ << "] : "
			  << "IMAGE FRAME"
			  << "\n\tdelta x top    = " << (Asy * (frame_height_ / 2) + Bsy) * (xrt - xlt)
			  << "\n\tdelta x bottom = " << (Asy * frame_height_ + Bsy) * (xrb - xlb)
			  << "\n\txmt_imgFrame   = " << xmt_imgFrame
			  << ", xmb_imgFrame = " << xmb_imgFrame
			  << "\n\troi_sy_ = " << roi_sy_
			  << ", frame_height_ = " << frame_height_
			  << std::endl;

	// Convert image Frame to car Frame
	// x image Frame maps into y car Frame
	// y image Frame maps into x car Frame

	// TOP point: (xmt_carFrame, ymt_carFrame)

	// img Frame : y = roi_sy_ and x = xmt_imgFrame
	// car Frame : y = xmt_imgFrame, x = calibration point measured in meters
	double xmt_carFrame = X_IMG_ROI_TOP_CAR_FRAME; // calibrated(measured) distance car CM to dash cam center in the groiund
	double ymt_carFrame = -xmt_imgFrame;
	// BOTTOM point:
	// img Frame : y = roi_sy_ + current_y_range_ and x = xmb_imgFrame
	// car Frame : y = xmb_imgFrame, x = calculated xmb_carFrame
	// Calculate bottom distance at the Car Frame
	// x car Frame at image center is 33 cm.
	// At any point near the car then the image center:
	// xmb_imgFrame = SUM(n =[0..y_range][-(Asy * (roi_sy_ + n_) + Bsy)];
	double xmb_carFrame = X_IMG_ROI_BOTTOM_CAR_FRAME;
	double ymb_carFrame = -xmb_imgFrame;

	// Calculate slope of the car direction
	double slope_carFrame = (ymt_carFrame - ymb_carFrame) / (xmt_carFrame - xmb_carFrame);
	// Calculate the intersect at the car Frame
	double intercept_carFrame = ymt_carFrame - slope_carFrame * xmt_carFrame;
	// Calculate the yaw angle
	double yaw_angle = std::atan(slope_carFrame); // in radians

	angle = static_cast<float>(yaw_angle); // Set the angle in radians
	offset = static_cast<float>(intercept_carFrame); // Set the offset in pmeters
	float angle_deg = angle * 180 / CV_PI;

    // Debugging output CAR FRAME
    std::cout << "[" << __func__ << "] : CAR FRAME"
				<< "\n\txmt_carFrame[" << xmt_carFrame << "], "
				<< "ymt_carFrame[" << ymt_carFrame << "], "
				<< "\n\txmb_carFrame[" << xmb_carFrame << "], "
				<< "ymb_carFrame[" << ymb_carFrame << "], "
				<< "\n\tslope car frame[" << slope_carFrame << "]"
				<< "\n\tyaw [" << angle * 180.0 / CV_PI << " deg], "
				<< "\n\tey  [" << offset << "]"
				<< std::endl;
	// Debugging output PIXEL
	std::cout << "[" << __func__ << "] PIXELS"
				<< "\n\tLeft  : slope = [" << iGeo_.left_slope << "] | intercept = [" << iGeo_.left_intercept << "]"
				<< "\n\tRight : slope = [" << iGeo_.right_slope << "] | intercept = [" << iGeo_.right_intercept << "]" << std::endl;
    std::cout << "[" << __func__ << "] : PIXELS"
				<< "\n\txlt[" << xlt << "], "
				<< "xmm[" << xmb << "], "
				<< "xrt[" << xrt << "], "
				<< "\n\txlb[" << xlb << "], "
				<< "xc [" << xc << "], "
				<< "xrb[" << xrb << "], "
				<< std::endl;
    // angle = angle_real; // Return the compensated true angle
}

void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle,
                                     float& smoothed_offset, float& smoothed_angle) {
    cv::Mat prediction = kf_.predict();
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured_offset, measured_angle);
    cv::Mat corrected = kf_.correct(measurement);
    smoothed_offset = corrected.at<float>(0);
    smoothed_angle = corrected.at<float>(1);
}

void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Error opening TensorRT model file: " + trt_model_path);
    }

    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr));
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create TensorRT execution context");

    cudaError_t err = cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float)); // 1x3x128x256
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float)); // 1x1x128x256
    if (err != cudaSuccess) {
        cudaFree(buffers_[0]);
        throw std::runtime_error("CUDA malloc failed for output buffer: " + std::string(cudaGetErrorString(err)));
    }

    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

void LaneDetector::infer() {
    cudaError_t err = cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to device failed: " + std::string(cudaGetErrorString(err)));

    if (!context_->enqueueV2(buffers_, stream_, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    err = cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) throw std::runtime_error("CUDA memcpy to host failed: " + std::string(cudaGetErrorString(err)));

    cudaStreamSynchronize(stream_);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA error after inference: " + std::string(cudaGetErrorString(err)));
}

void LaneDetector::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC); // Interpolação cúbica

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB); // Converte de BGR para RGB

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0); // Normaliza para [0,1]

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels); // Canais na ordem R, G, B
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

// yellow NOT processFrame():
void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

	// std::cout << "[" << __func__ << "] : "
	// 		<< "lane_mask_ size: " << lane_mask_.size()
	// 		<< ", Type: " << lane_mask_.type()
	// 		<< ", Channels: " << lane_mask_.channels() << std::endl;
	// Convert output data to cv::Mat
	// cv::Mat birdEyeMask = birdsEyeTransform(lane_mask_); // Transformação de perspectiva
    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    //cv::imwrite("lane_mask_original.png", lane_mask_ * 255);

    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);

	cv::Mat rawLane;
	cv::threshold(lane_mask_, rawLane, 0.5, 255.0, cv::THRESH_BINARY);
	rawLane.convertTo(rawLane, CV_8U);
	if (!rawLane.empty()) {
		cv::imwrite("rawLane.png", rawLane );
	}

    // double min_val, max_val;
    // cv::minMaxLoc(lane_mask_, &min_val, &max_val);
    // Do not remove the comment below, it is useful for debugging
    // std::cout << "["<<__func__<< "] : lane_mask_ min: " << min_val << ", max: " << max_val << std::endl;

    // cv::Mat gray;
    // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // cv::Scalar mean_intensity = cv::mean(gray);
    // float brightness = mean_intensity[0];
    // float threshold = brightness < 100 ? 0.1 : 0.3; // Even lower for yellow
    // Do not remove the comment below, it is useful for debugging
	// std::cout << "["<<__func__<< "] : Brightness: " << brightness << ", Threshold: " << threshold << std::endl;
    // cv::Mat binary_mask;
    float threshold = 0.5; // Limiar fixo, equivalente a (preds > 0.5).float()
    // cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);


    cv::Mat binary_mask;
    cv::threshold(lane_mask_, binary_mask, threshold, 1.0, cv::THRESH_BINARY);

    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Larger kernel
    // cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_DILATE, kernel);
    // cv::morphologyEx(binary_mask, lane_mask_, cv::MORPH_CLOSE, kernel);

    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_CUBIC); // Resize to original frame size

    output_frame = frame.clone();


    if (!calculateLaneGeometry(offset, angle)) {
        std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
    }

    debug_->showOutputVideo(rawLane, output_frame, iGeo_, CAMERA_OFFSET);

	// cv::Mat lane_mask_8u;
	// lane_mask_.convertTo(lane_mask_8u, CV_8U, 255.0);
    // cv::imwrite("lane_mask.png", lane_mask_ * 255);
    // cv::imwrite("binary_mask.png", binary_mask * 255);
}