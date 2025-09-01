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
#include "ObjectDetector.hpp"
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
    void autonomous(float ey, float yaw);
    void setLaneDetector(std::unique_ptr<LaneDetector> detector);
    void setObjectDetector(std::unique_ptr<ObjectDetector> detector);
    bool initialize();

    cv::VideoCapture cap_;

private:
    SDL_Joystick* joystick;
    JetCar* jetCar;
    std::unique_ptr<LaneDetector> laneDetector;
    std::unique_ptr<ObjectDetector> objectDetector;
    SpeedSubscriber speed;
    SpeedPIDController* speedPIDController;
    std::array<bool, 16> buttonStates;
    std::unordered_map<int, Actions> buttonActions;
    std::unordered_map<int, std::function<void(int)>> axisActions;
    std::atomic<float> currentSpeed;
    cv::VideoWriter video_writer;
    cv::Mat frame, output_frame;

	// State variables
    // Vector3d current_state_;
    TimeTracker tracker;
    int _currentMode;
	bool visualize_mask_{true};

    // CSV logging
    std::ofstream csv_file_;
    std::mutex csv_mutex_;
	float delta_;
	MPCController mpc_;

    // Stop control
    int stopCounter = 0;  // conta quantos frames seguidos detectou STOP
    const int STOP_THRESHOLD = 5;  // precisa de 5 frames seguidos pra acionar
    void checkStopSign(const std::vector<Detection>& detections);

	int cruise_speed_;

};


#endif