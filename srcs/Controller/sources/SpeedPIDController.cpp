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

    std::cout << "Current Speed: " << v_current << " m/s, Target Speed: " << v_target << " m/s" << std::endl;
    float error = v_target - v_current;
    std::cout << "Error: " << error << std::endl;
    std::cout << "dt: " << dt << std::endl;

    integral_ = integral_ + error * dt;
    std::cout << "Integral: " << integral_ << std::endl;
    float derivative = (error - prev_error_) / dt;
    std::cout << "Derivative: " << derivative << std::endl;
    prev_error_ = error;
    std::cout << "Error: " << error << ", Integral: " << integral_ << ", Derivative: " << derivative << std::endl;
    float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    std::cout << "PID Output: " << output << std::endl;
    return std::clamp(output, pwm_min_, pwm_max_);
}

float get_velocity(std::atomic<float>& currentSpeed) {
    return currentSpeed.load(std::memory_order_relaxed);
}
