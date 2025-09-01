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
#include "JoypadMapping.hpp"
#include "TimeTracker.hpp"
#include "SpeedPIDController.hpp"
#include "MPC.hpp"

class Controller {
public:

    struct State {
        float x, y, theta, v;
    };

    Controller();
    ~Controller();
    void processEvent(const SDL_Event& event);
    void listen();
    void autonomous(float ey, float yaw);
    void setLaneDetector(std::unique_ptr<LaneDetector> detector);
    void setObjectDetector(std::unique_ptr<ObjectDetector> detector);
    bool initialize();

    cv::VideoCapture cap_;

private:
    SDL_Joystick* joystick;
    JetCar jetCar;
    JoypadMapping mapping;
    std::unique_ptr<LaneDetector> laneDetector;
    std::unique_ptr<ObjectDetector> objectDetector;
    SpeedSubscriber speed;
    SpeedPIDController* speedPIDController;
    std::atomic<float> currentSpeed;    
    cv::VideoWriter video_writer;
    cv::Mat frame, output_frame;

	// State variables
    // Vector3d current_state_;
    TimeTracker tracker;
	bool visualize_mask_{true};

    // CSV logging
    std::ofstream csv_file_;
    std::mutex csv_mutex_;
	float delta_;
	MPCController mpc_;

    // Stop control
    int stopCounter = 0;  // conta quantos frames seguidos detectou STOP
    const int STOP_THRESHOLD = 5;  // precisa de 5 frames seguidos pra acionar
    bool checkStopSign(const std::vector<Detection>& detections);


};


#endif