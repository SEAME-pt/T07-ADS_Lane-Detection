#include <Controller.hpp>
#include "SpeedSubscriber.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

Controller::Controller(JetCar* jetCar) : joystick(nullptr), jetCar(jetCar), _currentMode(MODE_JOYSTICK) {
    // Initialize SDL for joystick input
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 Joystick: " + std::string(SDL_GetError()));
    }

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
    std::string pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
                          "rtph264pay ! udpsink host=239.255.0.1 port=5000 sync=false";
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
    while (true) {
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        if (_currentMode == MODE_AUTONOMOUS) {
			//delta must contain last value from servor motor
			delta_ = jetCar->get_servo_angle();
            autonomous(delta_);
        }

        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick connected, quitting..." << std::endl;
            break;
        }

        SDL_Delay(10);  // Small delay to avoid overloading CPU
    }
}

// Controller::State Controller::kinematicModel(const State& state, float delta, float a) {
//     // Unchanged from original
//     State next;
//     next.x = state.x + state.v * std::cos(state.theta) * DT;
//     next.y = state.y + state.v * std::sin(state.theta) * DT;
//     next.theta = state.theta + (state.v / L) * std::tan(delta) * DT;
//     next.v = state.v + a * DT;
//     return next;
// }

// void Controller::setupCostFunction(Eigen::MatrixXd& H, Eigen::VectorXd& f, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref) {
//     // Unchanged from original
//     H.setZero();
//     f.setZero();

//     for (int i = 0; i < N; ++i) {
//         H(i, i) = Q_y;                // Penalize lateral offset
//         H(N + i, N + i) = Q_theta;    // Penalize heading error
//         H(2 * N + i, 2 * N + i) = R_delta;  // Penalize steering effort
//         H(3 * N + i, 3 * N + i) = R_a;      // Penalize acceleration effort
//         f(i) = -Q_y * y_ref[i];             // Linear term for offset
//         f(N + i) = -Q_theta * theta_ref[i]; // Linear term for yaw
//     }

//     for (int i = 1; i < N; ++i) {
//         H(2 * N + i, 2 * N + i) += R_d_delta;
//         H(2 * N + i - 1, 2 * N + i - 1) += R_d_delta;
//     }
// }

// Eigen::VectorXd Controller::solveMPC(const State& initial_state, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref) {
//     // Unchanged from original
//     int n_vars = 4 * N;
//     Eigen::MatrixXd H(n_vars, n_vars);
//     Eigen::VectorXd f(n_vars);

//     setupCostFunction(H, f, y_ref, theta_ref);

//     Eigen::VectorXd control_sequence(2);
//     float steering = (0.5f * y_ref[0]) + (2.0f * theta_ref.mean());
//     control_sequence[0] = steering;

//     float base_throttle = 0.8f;
//     float throttle = base_throttle - 0.05f * fabs(theta_ref.mean());
//     control_sequence[1] = throttle;

//     return control_sequence;
// }

void Controller::autonomous(float prev_delta) {
	static float prev_offset = 0.0f; // Current offset of the vehicle
    static float prev_yaw = 0.0f;  // Store previous angle for rate calculation
	static float prev_speed = 1.0f;  // Current speed of the vehicle
	// static float prev_delta = 0.0f;  // Current speed of the delta
	Vector3d current_state_(prev_offset, prev_yaw, prev_speed); // Initialize current state

    // Check if LaneDetector is initialized and capture frame
    if (!laneDetector || !laneDetector->cap_.read(frame)) {
		if (!laneDetector) {
			std::cerr << "Error: LaneDetector not initialized!" << std::endl;
		} else {
			std::cerr << "Error: Could not read frame from camera!" << std::endl;
		}
       // std::cerr << "Error: Could not capture frame or LaneDetector not initialized!" << std::endl;
        return;
    }

    float offset, yaw;
    tracker.mark();
    laneDetector->processFrame(frame, offset, yaw, output_frame, true);

    // Calculate rate of change of yaw to predict curve
    float yaw_rate = (yaw - prev_yaw) / DT;  // deg/s
    prev_yaw = yaw;

    // Convert to MPC inputs
	// ?? ?????? acho que ja esta feito
    // float y_ref = offset;// * (1.0f / 640.0f);  // Convert pixels to meters (adjust scale if needed)
    float theta_ref = -yaw; // * (CV_PI / 180.0f);  // Invert yaw to correct for possible detection error

    // // Predict future trajectory over horizon with dynamic offset
    // Eigen::VectorXd y_ref_vec(N);
    // Eigen::VectorXd theta_ref_vec(N);
    // for (int i = 0; i < N; ++i) {
		//     float t = i * DT;
		//     // Extrapolate offset based on current offset and yaw (assuming constant speed and curvature)
		//     float dy = (current_state_.v * t * std::sin(theta_ref)) / 640.0f;  // Approximate lateral shift in meters
		//     y_ref_vec[i] = y_ref + dy;  // Update offset over time
		//     theta_ref_vec[i] = theta_ref + (yaw_rate * (CV_PI / 180.0f) * t);  // Linear extrapolation of yaw
		// }
	double delta = prev_delta;

    // Solve MPC to get control inputs
    Vector2d control = mpc_.solve_mpc(current_state_, delta);
    delta = control[0];  // Steering yaw (radians)
    float a = control[1];      // Acceleration (m/s²)

    // Apply constraints
    float steering = std::max(-DELTA_MAX, std::min(DELTA_MAX, delta));  // Limit to ±30 deg in radians
    //std::cout << "Final steering calculation " << (steering * (180.0f / CV_PI)) << std::endl;
    //jetCar->set_servo_angle(static_cast<int>(steering * (180.0f / CV_PI)));  // Convert radians to degrees
	std::cout << "Servor yaw: " << static_cast<int>(steering * (180.0f / CV_PI)) << std::endl;

	//float velocidade = velocidade + a * DT;  // Update speed based on acceleration
    // Update vehicle state
    //current_state_ = kinematicModel(current_state_, steering, a);
	current_state_ = mpc_.dynamics(current_state_, control);

	// Update speed in JetCar
	//prev_speed = currentSpeed.load(std::memory_order_relaxed);
	//currentSpeed.store(velocidade, std::memory_order_relaxed);

	// jetCar->set_motor_speed(static_cast<int>(current_state_(2) * 100.0f / 2.8f));  // Convert m/s to cm/s
	jetCar->set_servo_angle(current_state_(0) * (180.0f / CV_PI));  // Convert radians to degrees

	// Log current speed
	float velocidade = current_state_(2);  // Current speed in m/s
	prev_speed = velocidade;  // Update previous speed for next iteration
	currentSpeed.store(velocidade, std::memory_order_relaxed);

	// Debug output
	std::cout << "Offset: " << offset << " m, Yaw: " << yaw * (180.0f / CV_PI) << " deg, Speed: " << velocidade << " m/s" << std::endl;
	// std::cout << "motor speed: " << static_cast<int>(velocidade * 100.0f / 2.8f) << std::endl;

    // Log data to CSV (unchanged)
    auto now = std::chrono::system_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lock(csv_mutex_);
        csv_file_ << timestamp_ms << ","
                  << std::fixed << std::setprecision(2) << currentSpeed.load(std::memory_order_relaxed) << ","
                  << (steering * 180.0f / CV_PI) << ","
                  << yaw << ","
                  << offset << "\n";
        csv_file_.flush();
    }

    tracker.mark();
	std::string servo_text = "Servo: " + std::to_string(steering * 180.0 / CV_PI) + " deg";
	std::string delta_text = "Delta: " + std::to_string(delta) + " rad";
	std::string speed_text = "Speed: " + std::to_string(velocidade) + " m";
	cv::putText(output_frame, servo_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	cv::putText(output_frame, delta_text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	cv::putText(output_frame, speed_text, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    video_writer.write(output_frame);
}
void Controller::setLaneDetector(std::unique_ptr<LaneDetector> detector) {
    laneDetector = std::move(detector);
}