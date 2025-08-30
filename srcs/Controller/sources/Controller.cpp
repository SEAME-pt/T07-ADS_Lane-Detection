#include "Controller.hpp"
#include "SpeedSubscriber.hpp"
#include "MPC.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

Controller::Controller(JetCar* jetCar) : joystick(nullptr), jetCar(jetCar), _currentMode(MODE_JOYSTICK), mpc_(L, DT, N) {


	// Initialize SDL for joystick input
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 Joystick: " + std::string(SDL_GetError()));
    }

	// visualize_mask_ = true;
    // Initialize speedController
    speedPIDController = new SpeedPIDController();

    int joystickCount = SDL_NumJoysticks();
    std::cout << "Number of joysticks connected: " << joystickCount << std::endl;

    buttonStates.fill(false);  // Initialize all button states to false

    if (joystickCount > 0) {
        joystick = SDL_JoystickOpen(0);
        if (joystick) {
            std::cout << "Joystick 0 connected!" << std::endl;
        } else {
            throw std::runtime_error("Failed to open joystick: " + std::string(SDL_GetError()));
        }
    } else {
        throw std::runtime_error("No joystick detected.");
    }

    speed.start([this](float speed) {
        currentSpeed.store(speed, std::memory_order_relaxed);
    });

    // Setup video streaming pipeline
    // std::string pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
    //                       "rtph264pay ! udpsink host=239.255.0.1 port=5000 sync=false multi-cast=true";

std::string pipeline =
    "appsrc ! videoconvert ! video/x-raw,format=I420 ! "  // Force 4:2:0
    "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=239.255.0.1 port=5000 auto-multicast=true loop=1";




	video_writer.open(pipeline, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(640, 360), true);
    if (!video_writer.isOpened()) {
        throw std::runtime_error("Failed to open VideoWriter for streaming!");
    }
    std::cout << "Streaming started at udp://0.0.0.0:5000" << std::endl;
    std::cout << '<gst-launch-1.0 -v udpsrc udpsrc address=239.255.0.1 port=5000 caps="application/x-rtp, payload=96, encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false< std::endl;' << std::endl;

    // Initialize CSV file
    // csv_file_.open("lane_detection_log.csv", std::ios::out | std::ios::app);
    // if (!csv_file_.is_open()) {
    //     throw std::runtime_error("Failed to open CSV file for writing!");
    // }
    // Write CSV header
    // csv_file_ << "Timestamp,CurrentSpeed,SteeringAngle,LaneAngle,Offset,ImageFilename\n";
}

Controller::~Controller() {
    if (joystick) {
        SDL_JoystickClose(joystick);
    }
    SDL_Quit();
    // Close CSV file
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
}

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
	std::cout << "[" << __func__ << "] "
				<< "Camera pipeline opened successfully: \n"
				<< pipeline << std::endl;
	std::cout << "[" << __func__ << "] "
				<< "LaneDetector initialization concluded!"
				<< std::endl;
    return cap_.isOpened();
}

void Controller::setButtonAction(int button, Actions actions) {
    buttonActions[button] = actions;
}

void Controller::setAxisAction(int axis, std::function<void(int)> action) {
    axisActions[axis] = action;
}

void Controller::processEvent(const SDL_Event& event) {
    // Unchanged from original
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
        bool isPressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;

        if (button < static_cast<int>(buttonStates.size())) {
            // std::cout << "Button " << button << " " << (isPressed ? "pressed" : "released") << std::endl;
            buttonStates[button] = isPressed;
            if (buttonActions.find(button) != buttonActions.end()) {
                if (isPressed && buttonActions[button].onPress) {
                    buttonActions[button].onPress();
                } else if (!isPressed && buttonActions[button].onRelease) {
                    buttonActions[button].onRelease();
                }
            }
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode != MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        // std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode == MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        // std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end() && axis == 3) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYDEVICEADDED) {
        std::cout << "Joystick connected!" << std::endl;
        if (!joystick) {
            joystick = SDL_JoystickOpen(0);
            if (joystick) {
                std::cout << "Joystick 0 connected!" << std::endl;
            } else {
                throw std::runtime_error("Failed to open joystick: " + std::string(SDL_GetError()));
            }
        }
    } else if (event.type == SDL_JOYDEVICEREMOVED) {
        std::cout << "Joystick disconnected!" << std::endl;
        if (joystick) {
            SDL_JoystickClose(joystick);
            joystick = nullptr;
        }
        exit(1);
    }
}

void Controller::setMode(const int &mode) {
    _currentMode = mode;
}

int Controller::getMode() {
    return _currentMode;
}

void Controller::listen() {
    SDL_Event event;
	std::cout <<"[" << __func__ << "] "
				<< "\n\t** KALMAN : " << KALMAN
				<< " ** CAR CM : " << CAR_CM
				<< "\n\t## Q_EY : " << Q_EY
				<< " ## Q_YAW : " << Q_YAW
				<< " ## Q_V : " << Q_V
				<< "\n\t@@ V_REF_PWM : " << V_REF_PWM << std::endl;

    while (true) {
		auto loop_start = std::chrono::steady_clock::now();

        if (!cap_.read(frame) || frame.empty()) {
            std::cerr << "Fail to obtain frame!" << std::endl;
            continue;
        }

        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        std::vector<Detection> detections = objectDetector->infer(frame);
        // Desenhar caixas no output_frame
        for (const auto& det : detections) {
            cv::rectangle(output_frame, det.bbox, cv::Scalar(0, 255, 0), 2);
            std::string label = det.class_name + " " + std::to_string(int(det.confidence*100)) + "%";
            cv::putText(output_frame, label, cv::Point(det.bbox.x, det.bbox.y-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        float ey, yaw;
        laneDetector->processFrame(frame, ey, yaw, output_frame, visualize_mask_);
        if (_currentMode == MODE_AUTONOMOUS) {
			if (currentSpeed.load(std::memory_order_relaxed) < V_MIN) {
				jetCar->set_motor_speed(static_cast<int>(V_REF_PWM));  // Convert m/s to cm/s
			}
			autonomous(ey, yaw);
			visualize_mask_ = false;
		} else {
			visualize_mask_ = true;
		}

        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick connected, quitting..." << std::endl;
            break;
        }

        // Display mode on output_frame bottom right corner
        std::string modeText = (_currentMode == MODE_JOYSTICK) ? "Joystick Mode" : "Autonomous Mode";
        cv::putText(output_frame, modeText, cv::Point(330, 340), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        video_writer.write(output_frame);

        // SDL_Delay(10);  // Small delay to avoid overloading CPU

		// End timing
		auto loop_end = std::chrono::steady_clock::now();
		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

		// std::cout << "[" << __func__ << "] "
		// 		<< "Loop duration: " << duration_ms << " ms \r" << std::flush;
		if (duration_ms < 99) {
        	std::this_thread::sleep_for(std::chrono::milliseconds(100 - duration_ms));
    	}

		// // Optional: Print actual duration (will be ~100+ ms)
		// auto loop_total_end = std::chrono::steady_clock::now();
		// auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_total_end - loop_start).count();
		// std::cout << "Loop duration (with delay): " << total_duration << " ms" << std::endl;

	}
}



void Controller::autonomous(float ey, float yaw) {
    tracker.mark();

	float speed = currentSpeed.load(std::memory_order_relaxed);  // Get current speed from SpeedSubscriber
	if (speed > 1000 || speed < -1000) {
		speed = 0.0f;  // Reset speed if it exceeds a threshold
	}
	if (speed > V_MIN ) {
		// std::cout << "[" << __func__ << "] Speed is " << speed << ", LKAS ON" << std::endl;
		mpc_.update(-ey, -yaw, speed);
	} else {
		std::cout << "[" << __func__ << "] Speed is " << speed <<  "! Too low, LKAS OFF" << std::endl;
		return;
	}

	float delta = 1.0f * mpc_.getSteeringAngle();
	float a = mpc_.getAcceleration();

	float steeringDEG = static_cast<int>(std::max(-DELTA_MAX, std::min(DELTA_MAX, static_cast<double>(delta))) * 180 / CV_PI);  // Convert radians to % PWM
	jetCar->set_servo_angle(static_cast<int>(steeringDEG));  // Converted to degrees

    tracker.mark();
	std::string servo_text = "Servo: " + std::to_string(delta * 180.0 / CV_PI) + " deg";
	std::string delta_text = "Delta: " + std::to_string(delta) + " rad";
	std::string speed_text = "Speed: " + std::to_string(speed) + " m/s";
	cv::putText(output_frame, servo_text, cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
	cv::putText(output_frame, delta_text, cv::Point(10, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
	cv::putText(output_frame, speed_text, cv::Point(10, 330), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

}

void Controller::setLaneDetector(std::unique_ptr<LaneDetector> detector) {
    laneDetector = std::move(detector);
}

void Controller::setObjectDetector(std::unique_ptr<ObjectDetector> detector) {
	objectDetector = std::move(detector);
}