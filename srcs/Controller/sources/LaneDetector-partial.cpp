#include "LaneDetector.hpp"
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);
    kf_ = cv::KalmanFilter(2, 2, 0, CV_32F);
    kf_.statePre.at<float>(0) = 0.0f;
    kf_.statePre.at<float>(1) = 0.0f;
    kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPre, cv::Scalar::all(1));

    input_height_ = 128;
    input_width_ = 256;
    frame_height_ = 360;
    frame_width_ = 640;
    roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT);
    roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);
    roi_sx_ = ROI_X_BORDER;
    roi_ex_ = frame_width_ - ROI_X_BORDER;
    roi_w_ = roi_ex_ - roi_sx_;
    roi_h_ = roi_ey_ - roi_sy_;
    imgFrame_.xcPX = frame_width_ / 2.0f + CAMERA_OFFSET;

    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = -1.0f; // Initialize to invalid value (meters)
    prev_left_edge_ = frame_width_ / 2;
    prev_right_edge_ = frame_width_ / 2;
    last_left_edge_ = frame_width_ / 2;
    last_right_edge_ = frame_width_ / 2;

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
    std::string pipeline = "nvarguscamerasrc exposuretimerange=\"1000000 50000000\" gainrange=\"1 16\" !"
                          "video/x-raw(memory:NVMM), width=640, height=360, "
                          "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                          "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera pipeline!" << std::endl;
        return false;
    }
    std::cout << "[" << __func__ << "] Camera pipeline opened successfully: \n" << pipeline << std::endl;
    std::cout << "[" << __func__ << "] LaneDetector initialization concluded!" << std::endl;
    return cap_.isOpened();
}

/// @brief Calculate lane geometry based on detected edges.
/// This function estimates the offset and angle of the lane based on the detected left and right edges.
/// It uses the slopes and intercepts of the detected edges to compute the lane geometry.
/// @param offset	The offset from the center of the lane in meters, in the car frame system of coordinates.
/// @param angle	The angle of the lane in radians, in the car frame system of coordinates.
/// @return 		True if lane geometry was successfully calculated, false otherwise.
void LaneDetector::defineROI() {
    std::cout << "[" << __func__ << "] Started..." << std::endl;
    roi_sy_ = static_cast<int>(frame_height_ * ROI_SY_PERCENT);
    roi_ey_ = static_cast<int>(frame_height_ * ROI_EY_PERCENT);
    // Dynamic ROI adjustment based on angle in radians
    int shift = static_cast<int>(iGeo_.angle * 114.6f); // ~2 pixels per degree (1 radian ≈ 57.3 degrees)
    roi_sx_ = std::max(0, ROI_X_BORDER - shift);
    roi_ex_ = std::min(frame_width_, frame_width_ - ROI_X_BORDER - shift);
    roi_w_ = roi_ex_ - roi_sx_;
    roi_h_ = roi_ey_ - roi_sy_;
    std::cout << "[" << __func__ << "] Ready ROI" << std::endl;
}

/// @brief 				Create left and right edges vectors from lane mask.
/// This function scans the lane mask within the specified ROI to find the left and right edges of the lane.
/// It uses a dense sampling approach to identify the first detected lane pixel in each row.
/// @note 				The function clears the left_edges_ and right_edges_ vectors before populating them.
/// @param lane_mask 	The lane mask image containing the detected lane pixels.
/// @param roi			 The region of interest (ROI) within the lane mask to search for edges.
/// @return 			True if enough edge points were found, false otherwise.
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
    }

    return left_edges_.size() >= MIN_EDGE_POINTS || right_edges_.size() >= MIN_EDGE_POINTS;
}

/// @brief 	Calculate the weighted linear regression for the detected edges.
/// This function computes the slope and intercept of the line that best fits
/// the detected edges using a weighted linear regression.
/// @note 	The function uses a weighted approach where the weight is based
/// on the vertical position of the edge points.
/// @param edges		The vector of edge points detected in the lane mask.
/// @param slope		The slope of the fitted line.
/// @param intercept 	The intercept of the fitted line.
void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& edges, double& slope, double& intercept) {
    if (edges.size() < MIN_EDGE_POINTS) {
        slope = 0.0;
        intercept = frame_width_ / 2.0 - CAMERA_OFFSET;
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int y_start = static_cast<int>(edges[0].y);
    int y_end = static_cast<int>(y_start + edges.size());
    int y_range = y_end - y_start;

    for (const auto& pt : edges) {
        double y = pt.y;
        double x = pt.x;
        double weight = (y - y_start) / static_cast<double>(y_range);
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

/// @brief Calculate the middle lane line based on the detected edges.
/// This function calculates the left and right edges of the lane at the top and bottom of the
/// image frame, and computes the midpoints at the top and bottom.
/// It uses the slopes and intercepts of the detected edges to compute the midpoints.
/// It also calculates the lane width based on the detected edges.
/// @note This function assumes that the left and right edges have been detected and stored in the
/// `left_edges_` and `right_edges_` vectors.
/// It updates the `imgFrame_` structure with the calculated midpoints and lane width.
/// If any edge is missing, it estimates the missing edge based on the available edge and the estimated lane width.
/// @return 	True if the middle lane line was successfully calculated, false otherwise.
bool LaneDetector::calculateMiddleLaneLine() {
    bool left_valid = left_edges_.size() >= MIN_EDGE_POINTS;
    bool right_valid = right_edges_.size() >= MIN_EDGE_POINTS;

    float lane_width_meters = estimated_lane_width_ < 0.0f ? 0.5f : estimated_lane_width_; // Fallback to 0.5m

    // Compute edge points in pixel frame
    if (left_valid) {
        imgFrame_.xlbPX = iGeo_.left_slope * frame_height_ + iGeo_.left_intercept;
        imgFrame_.xltPX = iGeo_.left_slope * (frame_height_ / 2.0f) + iGeo_.left_intercept;
    }
    if (right_valid) {
        imgFrame_.xrbPX = iGeo_.right_slope * frame_height_ + iGeo_.right_intercept;
        imgFrame_.xrtPX = iGeo_.right_slope * (frame_height_ / 2.0f) + iGeo_.right_intercept;
    }

    // Update lane width (meters) when both edges are valid
    if (left_valid && right_valid) {
        float lane_width_pixels = 0.0f;
        // Use fitted lines at ROI bottom (y = frame_height_) and middle (y = frame_height_/2)
        float xlb = imgFrame_.xlbPX;
        float xrb = imgFrame_.xrbPX;
        float xlt = imgFrame_.xltPX;
        float xrt = imgFrame_.xrtPX;
        lane_width_pixels = ((xrb - xlb) + (xrt - xlt)) / 2.0f;
        float lane_width_meters_new = (Asy * frame_height_ + Bsy) * lane_width_pixels;
        lane_width_history_.push_back(lane_width_meters_new);
        if (lane_width_history_.size() > MAX_HISTORY_SIZE) {
            lane_width_history_.erase(lane_width_history_.begin());
        }
        estimated_lane_width_ = std::accumulate(lane_width_history_.begin(), lane_width_history_.end(), 0.0f) / lane_width_history_.size();
        iGeo_.lane_width = estimated_lane_width_;
        lane_width_meters = estimated_lane_width_; // Update for this frame
    } else if (left_valid && !right_valid) {
        // Case 2: Yes/No - Estimate right edge in car frame
        float xlb_img = (Asy * frame_height_ + Bsy) * (imgFrame_.xlbPX - imgFrame_.xcPX);
        float xlt_img = (Asy * (frame_height_ / 2.0f) + Bsy) * (imgFrame_.xltPX - imgFrame_.xcPX);
        float ylb_car = -xlb_img;
        float ylt_car = -xlt_img;
        float slope_car = (ylt_car - ylb_car) / (X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM);
        float intercept_car = ylb_car - slope_car * X_CAR_FRAME_BOTTOM;

        float yrb_car = ylb_car + lane_width_meters;
        float yrt_car = ylt_car + lane_width_meters;
        float xrb_img = -yrb_car;
        float xrt_img = -yrt_car;
        imgFrame_.xrbPX = xrb_img / (Asy * frame_height_ + Bsy) + imgFrame_.xcPX;
        imgFrame_.xrtPX = xrt_img / (Asy * (frame_height_ / 2.0f) + Bsy) + imgFrame_.xcPX;
        iGeo_.right_slope = (imgFrame_.xrtPX - imgFrame_.xrbPX) / ((frame_height_ / 2.0f) - frame_height_);
        iGeo_.right_intercept = imgFrame_.xrbPX - iGeo_.right_slope * frame_height_;
        if (estimated_lane_width_ > 0.0f) {
            float lane_width_pixels = imgFrame_.xrbPX - imgFrame_.xlbPX;
            float lane_width_meters_new = (Asy * frame_height_ + Bsy) * lane_width_pixels;
            lane_width_history_.push_back(lane_width_meters_new);
            if (lane_width_history_.size() > MAX_HISTORY_SIZE) {
                lane_width_history_.erase(lane_width_history_.begin());
            }
            estimated_lane_width_ = std::accumulate(lane_width_history_.begin(), lane_width_history_.end(), 0.0f) / lane_width_history_.size();
            iGeo_.lane_width = estimated_lane_width_;
        }

    } else if (!left_valid && right_valid) {
        // Case 3: No/Yes - Estimate left edge in car frame
        float xrb_img = (Asy * frame_height_ + Bsy) * (imgFrame_.xrbPX - imgFrame_.xcPX);
        float xrt_img = (Asy * (frame_height_ / 2.0f) + Bsy) * (imgFrame_.xrtPX - imgFrame_.xcPX);
        float yrb_car = -xrb_img;
        float yrt_car = -xrt_img;
        float slope_car = (yrt_car - yrb_car) / (X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM);
        float intercept_car = yrb_car - slope_car * X_CAR_FRAME_BOTTOM;

        float ylb_car = yrb_car - lane_width_meters;
        float ylt_car = yrt_car - lane_width_meters;
        float xlb_img = -ylb_car;
        float xlt_img = -ylt_car;
        imgFrame_.xlbPX = xlb_img / (Asy * frame_height_ + Bsy) + imgFrame_.xcPX;
        imgFrame_.xltPX = xlt_img / (Asy * (frame_height_ / 2.0f) + Bsy) + imgFrame_.xcPX;
        iGeo_.left_slope = (imgFrame_.xltPX - imgFrame_.xlbPX) / ((frame_height_ / 2.0f) - frame_height_);
        iGeo_.left_intercept = imgFrame_.xlbPX - iGeo_.left_slope * frame_height_;
        if (estimated_lane_width_ > 0.0f) {
            float lane_width_pixels = imgFrame_.xrbPX - imgFrame_.xlbPX;
            float lane_width_meters_new = (Asy * frame_height_ + Bsy) * lane_width_pixels;
            lane_width_history_.push_back(lane_width_meters_new);
            if (lane_width_history_.size() > MAX_HISTORY_SIZE) {
                lane_width_history_.erase(lane_width_history_.begin());
            }
            estimated_lane_width_ = std::accumulate(lane_width_history_.begin(), lane_width_history_.end(), 0.0f) / lane_width_history_.size();
            iGeo_.lane_width = estimated_lane_width_;
        }
    } else {
        // Case 4: No/No
        return false;
    }

    // Compute midpoints in pixel frame
    imgFrame_.xmbPX = (imgFrame_.xlbPX + imgFrame_.xrbPX) / 2.0f;
    imgFrame_.xmtPX = (imgFrame_.xltPX + imgFrame_.xrtPX) / 2.0f;

    // Convert to image frame (meters)
    imgFrame_.xmt = (Asy * (frame_height_ / 2.0f) + Bsy) * (imgFrame_.xmtPX - imgFrame_.xcPX);
    imgFrame_.xmb = (Asy * frame_height_ + Bsy) * (imgFrame_.xmbPX - imgFrame_.xcPX);

    // Debugging output
    std::cout << "[" << __func__ << "] : PIXELS"
              << "\n\t"
              << "xlt[" << imgFrame_.xltPX << "], "
              << "xmt[" << imgFrame_.xmtPX << "], "
              << "xrt[" << imgFrame_.xrtPX << "], "
              << "\n\t"
              << "xlb[" << imgFrame_.xlbPX << "], "
              << "xmb[" << imgFrame_.xmbPX << "], "
              << "xrb[" << imgFrame_.xrbPX << "], "
              << "\n\tlane_width[" << iGeo_.lane_width << "m]"
              << std::endl;

    std::cout << "[" << __func__ << "] : METERS"
              << "\n\t"
              << "xmt[" << imgFrame_.xmt << "], "
              << "xmb[" << imgFrame_.xmb << "], "
              << std::endl;

    return true;
}

/// @brief 	Calculate the offset and angle of the lane in the car frame.
/// This function converts the image frame midlane points from pixels to meters and calculates the offset and
/// angle of the lane in the car frame system of coordinates.
/// The offset is the distance from the center of the car to the lane, and the angle
/// is the yaw angle of the lane in radians.
/// The conversion uses a scale function defined by the slope (Asy) and intercept (Bsy) of the scale function.
/// @note 	The function assumes that the image frame midlane points have been calculated and stored in the `imgFrame_` structure.
/// @param offset 	The offset from the center of the car to the lane in meters.
/// @param angle 	The yaw angle of the lane in radians.
void LaneDetector::calculateOffsetAndAngle(float& offset, float& angle) const {
    carFrame_.yT = -imgFrame_.xmt;
    carFrame_.yB = -imgFrame_.xmb;

    carFrame_.slope = (carFrame_.yT - carFrame_.yB) / carFrame_.xDelta;
    carFrame_.intercept = carFrame_.yT - carFrame_.slope * carFrame_.xT;

    angle = static_cast<float>(std::atan(carFrame_.slope));
    offset = static_cast<float>(carFrame_.intercept);

    std::cout << "[" << __func__ << "] : CAR FRAME"
              << "\n\txmt_carFrame[" << carFrame_.xT << "], "
              << "ymt_carFrame[" << carFrame_.yT << "], "
              << "\n\txmb_carFrame[" << carFrame_.xB << "], "
              << "ymb_carFrame[" << carFrame_.yB << "], "
              << "\n\tslope car frame[" << carFrame_.slope << "]"
              << "\n\tyaw [" << angle << " rad], "
              << "\n\tey  [" << offset << "]"
              << std::endl;
    std::cout << "[" << __func__ << "] PIXELS"
              << "\n\tLeft  : slope = [" << iGeo_.left_slope << "] | intercept = [" << iGeo_.left_intercept << "]"
              << "\n\tRight : slope = [" << iGeo_.right_slope << "] | intercept = [" << iGeo_.right_intercept << "]" << std::endl;
}

/// @brief Calculate lane geometry based on detected edges.
/// This function estimates the offset and angle of the lane based on the detected left and right edges.
/// It uses the slopes and intercepts of the detected edges to compute the lane geometry.
/// @param offset	The offset from the center of the lane in meters, in the car frame system of coordinates.
/// @param angle	The angle of the lane in radians, in the car frame system of coordinates.
/// @return 		True if lane geometry was successfully calculated, false otherwise.
bool LaneDetector::calculateLaneGeometry(float& offset, float& angle) {
    if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
        std::cerr << "Invalid lane mask!" << std::endl;
        return false;
    }

    cv::Rect roi(roi_sx_, roi_sy_, roi_ex_ - roi_sx_, roi_ey_ - roi_sy_);
    if (roi.width <= 0 || roi.height <= 0) {
        std::cerr << "Invalid ROI dimensions!" << std::endl;
        return false;
    }

    if (!findLaneEdges(lane_mask_, roi)) {
        cv::Mat prediction = kf_.predict();
        offset = prediction.at<float>(0);
        angle = prediction.at<float>(1);
        iGeo_.offset = offset;
        iGeo_.angle = angle;
        return true;
    }

    bool left_valid = left_edges_.size() >= MIN_EDGE_POINTS;
    bool right_valid = right_edges_.size() >= MIN_EDGE_POINTS;

    if (left_valid) {
        weightedLinearRegression(left_edges_, iGeo_.left_slope, iGeo_.left_intercept);
    }
    if (right_valid) {
        weightedLinearRegression(right_edges_, iGeo_.right_slope, iGeo_.right_intercept);
    }

    if (!calculateMiddleLaneLine()) {
        cv::Mat prediction = kf_.predict();
        offset = prediction.at<float>(0);
        angle = prediction.at<float>(1);
        iGeo_.offset = offset;
        iGeo_.angle = angle;
        return true;
    }

    float measured_offset, measured_angle;
    calculateOffsetAndAngle(measured_offset, measured_angle);

    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    offset = smoothed_offset;
    angle = smoothed_angle;

    iGeo_.offset = offset;
    iGeo_.angle = angle;

    // Store only offset and angle in history
    imageGeometry history_entry;
    history_entry.offset = offset;
    history_entry.angle = angle;
    history_.push_back(history_entry);
    if (history_.size() > MAX_HISTORY_SIZE) {
        history_.erase(history_.begin());
    }

    return true;
}

void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle, float& smoothed_offset, float& smoothed_angle) {
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

    cudaError_t err = cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
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
    cv::resize(frame, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();
    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::Mat exp_mask;
    cv::exp(-lane_mask_, exp_mask);
    lane_mask_ = 1.0 / (1.0 + exp_mask);
    cv::Mat rawLane;
    cv::threshold(lane_mask_, rawLane, 0.5, 255.0, cv::THRESH_BINARY);
    rawLane.convertTo(rawLane, CV_8U);
    if (!rawLane.empty()) {
        cv::imwrite("rawLane.png", rawLane);
    }
    cv::Mat binary_mask;
    cv::threshold(lane_mask_, binary_mask, THRESHOLD, 1.0, cv::THRESH_BINARY);
    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_), 0, 0, cv::INTER_CUBIC);
    output_frame = frame.clone();
    if (!calculateLaneGeometry(offset, angle)) {
        std::cout << "[" << __func__ << "] Failed to calculate lane geometry" << std::endl;
    }
    debug_->showOutputVideo(rawLane, output_frame, iGeo_, CAMERA_OFFSET);
}