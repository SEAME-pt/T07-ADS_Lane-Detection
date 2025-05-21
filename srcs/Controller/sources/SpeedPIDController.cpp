// SpeedPIDController.cpp**
#include "SpeedPIDController.hpp"
#include <algorithm>
#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>

SpeedPIDController::SpeedPIDController()
    : prev_error_(0.0f), integral_(0.0f) {}

void SpeedPIDController::reset() {
    prev_error_ = 0.0f;
    integral_ = 0.0f;
}

float SpeedPIDController::update(float v_current, float v_target, float dt) {

    float error = v_target - v_current;


    integral_ = integral_ + error * dt;
    float derivative = (error - prev_error_) / dt;
    prev_error_ = error;
    float output = v_target + (kp_ * error + ki_ * integral_ + kd_ * derivative);
    return std::clamp(output, pwm_min_, pwm_max_);
}

float get_velocity(std::atomic<float>& currentSpeed) {
    return currentSpeed.load(std::memory_order_relaxed);
}
