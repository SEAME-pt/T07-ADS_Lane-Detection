#include "Controller.hpp"
#include "SpeedSubscriber.hpp"
#include "MPC.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <deque>

std::deque<cv::Rect> stopHistory;
const int HISTORY_SIZE = 8;
const int MIN_CONFIRM = 3;  // STOP needs to appear in at least 6 of the last 10 frames

Controller::Controller(JetCar* jetCar) : 
    joystick(nullptr), jetCar(jetCar), _currentMode(MODE_JOYSTICK), mpc_(L, DT, N) 
{
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 Joystick: " + std::string(SDL_GetError()));
    }

    speedPIDController = new SpeedPIDController();

    int joystickCount = SDL_NumJoysticks();
    std::cout << "Number of joysticks connected: " << joystickCount << std::endl;

    buttonStates.fill(false);

    if (joystickCount > 0) {
        joystick = SDL_JoystickOpen(0);
        if (joystick) std::cout << "Joystick 0 connected!" << std::endl;
        else throw std::runtime_error("Failed to open joystick: " + std::string(SDL_GetError()));
    } else {
        throw std::runtime_error("No joystick detected.");
    }

    speed.start([this](float speed) {
        currentSpeed.store(speed, std::memory_order_relaxed);
    });

    // std::string pipeline =
    //     "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
    //     "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
    //     "rtph264pay config-interval=1 pt=96 ! "
    //     "udpsink host=239.255.0.1 port=5000 auto-multicast=true loop=1";

    // video_writer.open(pipeline, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(640, 360), true);
    // if (!video_writer.isOpened())
    //     throw std::runtime_error("Failed to open VideoWriter for streaming!");
    
    std::cout << "Streaming started at udp://0.0.0.0:5000" << std::endl;
}

Controller::~Controller() {
    if (joystick) SDL_JoystickClose(joystick);
    SDL_Quit();
}

/**
 * @brief Initialize camera pipeline
 * @return True if camera opened successfully
 */
bool Controller::initialize() {
    std::string pipeline = "nvarguscamerasrc exposuretimerange=\"1000000 50000000\" gainrange=\"1 16\" !"
                           "video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";

    cap_.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera pipeline!" << std::endl;
        return false;
    }

    std::cout << "[" << __func__ << "] Camera pipeline opened successfully." << std::endl;
    return true;
}

void Controller::setButtonAction(int button, Actions actions) {
    buttonActions[button] = actions;
}

void Controller::setAxisAction(int axis, std::function<void(int)> action) {
    axisActions[axis] = action;
}

void Controller::processEvent(const SDL_Event& event) {
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
        bool isPressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;

        if (button < static_cast<int>(buttonStates.size())) {
            buttonStates[button] = isPressed;
            if (buttonActions.find(button) != buttonActions.end()) {
                if (isPressed && buttonActions[button].onPress) buttonActions[button].onPress();
                else if (!isPressed && buttonActions[button].onRelease) buttonActions[button].onRelease();
            }
        }
    } else if (event.type == SDL_JOYAXISMOTION) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYDEVICEADDED) {
        if (!joystick) joystick = SDL_JoystickOpen(0);
    } else if (event.type == SDL_JOYDEVICEREMOVED) {
        if (joystick) SDL_JoystickClose(joystick);
        joystick = nullptr;
        exit(1);
    }
}

void Controller::setMode(const int &mode) { _currentMode = mode; }
int Controller::getMode() { return _currentMode; }

/**
 * @brief Main loop for reading camera, running detections and controlling car.
 *        Object detection and lane detection are limited by frame skip thresholds.
 */
void Controller::listen() {
    SDL_Event event;
    const int OBJ_DET_SKIP = 1;   // run object detection every 3 frames
    const int LANE_DET_SKIP = 1;  // run lane detection every frame

    int objCounter = 0;
    int laneCounter = 0;

    float ey = 0.0f, yaw = 0.0f;

    while (true) {
        // auto loop_start = std::chrono::steady_clock::now();

        if (!cap_.read(frame) || frame.empty()) continue;
        output_frame = frame.clone();

        while (SDL_PollEvent(&event)) processEvent(event);

        // Object detection frame limiter
        //if (objCounter % OBJ_DET_SKIP == 0) {
            //}
            // objCounter++;

            // Lane detection frame limiter
            // if (laneCounter % LANE_DET_SKIP == 0) {
        laneDetector->processFrame(frame.clone(), ey, yaw, output_frame, visualize_mask_);
        //}
        //laneCounter++;
        
        // Autonomous control
        if (_currentMode == MODE_AUTONOMOUS) {
            if (currentSpeed.load(std::memory_order_relaxed) < V_MIN)
                jetCar->set_motor_speed(static_cast<int>(V_REF_PWM));
            autonomous(ey, yaw);
            visualize_mask_ = false;
        } else {
            visualize_mask_ = true;
        }
        
        cv::Rect roi(frame.cols * 0.6, 0, frame.cols * 0.4, frame.rows);
        std::vector<Detection> detections = objectDetector->infer(frame, roi);
        if (checkStopSign(detections)) {
            jetCar->set_motor_speed(0);
            setMode(MODE_JOYSTICK);
        }

        // Exit conditions
        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) break;
        if (!joystick) break;

        // std::string modeText = (_currentMode == MODE_JOYSTICK) ? "Joystick Mode" : "Autonomous Mode";
        // cv::putText(output_frame, modeText, cv::Point(330, 340), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255),1);
        // video_writer.write(output_frame);

        // auto loop_end = std::chrono::steady_clock::now();
        // auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        // std::cout << "[listen] Loop took " << duration_ms << " ms" << std::endl;

        // if (duration_ms < 100) std::this_thread::sleep_for(std::chrono::milliseconds(100 - duration_ms));
    }
}

/**
 * @brief Check for stop signs and confirm with historical frames
 * @param detections Object detection results
 * @return true if stop is confirmed
 */
bool Controller::checkStopSign(const std::vector<Detection>& detections) {
    bool found = false;
    cv::Rect currentBox;

    for (const auto& det : detections) {
        if ((det.class_name == "STOP" || det.class_name == "crosswalk") && det.confidence > 0.7) {
            found = true;
            currentBox = det.bbox;
            break;
        }
    }

    if (found) stopHistory.push_back(currentBox);
    else stopHistory.push_back(cv::Rect());

    if (stopHistory.size() > HISTORY_SIZE) stopHistory.pop_front();

    int count = 0;
    for (auto& b : stopHistory) if (b.area() > 0) count++;

    if (count >= MIN_CONFIRM) {
        cv::Rect ref = stopHistory.back();
        int consistent = 0;
        for (auto& b : stopHistory) {
            if (b.area() > 0) {
                float iou = (float)(ref & b).area() / (ref | b).area();
                if (iou > 0.3) consistent++;
            }
        }

        if (consistent >= MIN_CONFIRM/2 && ref.width > frame.cols * 0.05) {
            std::cout << "[STOP] Confirmed!" << std::endl;
            stopHistory.clear();
            return true;
        }
    }

    return false;
}

/**
 * @brief Run MPC and control servo/motor based on ey and yaw
 */
void Controller::autonomous(float ey, float yaw) {
    // tracker.mark();

    float speed = currentSpeed.load(std::memory_order_relaxed);
    if (speed > 1000 || speed < -1000) speed = 0.0f;

    if (speed > V_MIN) {
        mpc_.update(-ey, -yaw, speed);
    } else {
        std::cout << "[autonomous] Speed too low, LKAS OFF" << std::endl;
        return;
    }

    float delta = mpc_.getSteeringAngle();
    float steeringDEG = std::max(-DELTA_MAX, std::min(DELTA_MAX, (double)delta)) * 180.0 / CV_PI;
    jetCar->set_servo_angle(static_cast<int>(steeringDEG));

    // tracker.mark();
    // cv::putText(output_frame, "Servo: " + std::to_string(delta*180/CV_PI) + " deg", cv::Point(10,270), cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar(255,255,255),1);
}

/**
 * @brief Set lane detector
 */
void Controller::setLaneDetector(std::unique_ptr<LaneDetector> detector) {
    laneDetector = std::move(detector);
}

/**
 * @brief Set object detector
 */
void Controller::setObjectDetector(std::unique_ptr<ObjectDetector> detector) {
    objectDetector = std::move(detector);
}
