#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <SDL2/SDL.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <Eigen/Dense>
#include <functional>
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <fstream>
#include "JetCar.hpp"
#include "LaneDetector.hpp"
#include "SpeedSubscriber.hpp"
#include "TimeTracker.hpp"
#include "SpeedPIDController.hpp"
#include "MPC.hpp"

#define BTN_A 0
#define BTN_B 1
#define BTN_X 3
#define BTN_Y 4
#define BTN_LB 6
#define BTN_RB 7
#define BTN_SELECT 10
#define BTN_START 11
#define BTN_HOME 12
#define BTN_LSTICK 13
#define BTN_RSTICK 14

enum Mode {
    MODE_JOYSTICK,
    MODE_AUTONOMOUS
};

struct Actions {
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

class Controller {
public:

    struct State {
        float x, y, theta, v;
    };

    Controller(JetCar* jetCar);
    ~Controller();

    void setButtonAction(int button, Actions actions);
    void setAxisAction(int axis, std::function<void(int)> action);
    void processEvent(const SDL_Event& event);
    void setMode(const int &mode);
    int  getMode();
    void listen();
    void autonomous(float prev_delta);
    void setLaneDetector(std::unique_ptr<LaneDetector> detector);

private:
    SDL_Joystick* joystick;
    JetCar* jetCar;
    std::unique_ptr<LaneDetector> laneDetector;
    SpeedSubscriber speed;
    SpeedPIDController* speedPIDController;
    std::array<bool, 16> buttonStates;
    std::unordered_map<int, Actions> buttonActions;
    std::unordered_map<int, std::function<void(int)>> axisActions;
    std::atomic<float> currentSpeed;
    cv::VideoWriter video_writer;
    cv::Mat frame, output_frame;

	// State variables
    Vector3d current_state_;
    TimeTracker tracker;
    int _currentMode;

    // CSV logging
    std::ofstream csv_file_;
    std::mutex csv_mutex_;
	float delta_;
	MPC mpc_;
    // // MPC parameters
    // static constexpr int N = 10;  // Prediction horizon
    // static constexpr float DT = 0.03f;  // Time step
    // static constexpr float L = 0.15f;   // Wheelbase
    // static constexpr float MAX_DELTA = 0.52f;  // Max steering angle (radians) => 30graus
    // static constexpr float Q_y = 100.0f;      // Weight for lateral offset
    // static constexpr float Q_theta = 50.0f;   // Weight for heading error
    // static constexpr float R_delta = 10.0f;   // Weight for steering effort
    // static constexpr float R_a = 5.0f;        // Weight for acceleration effort
    // static constexpr float R_d_delta = 20.0f; // Weight for steering rate

    // State kinematicModel(const State& state, float delta, float a);
    // void setupCostFunction(Eigen::MatrixXd& H, Eigen::VectorXd& f, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref);
    // Eigen::VectorXd solveMPC(const State& initial_state, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref);

};


#endif