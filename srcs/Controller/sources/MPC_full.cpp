// MPC.cpp
// Version: v13 (2025-09-03) ADAPTATIVE MPC WITH ACCELERATION
#include "MPC-full.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

MPCController::MPCController(float wheelbase, float dt, int horizon, float v_cruise)
    : L_(wheelbase), dt_(dt), N_(horizon), k_ff_(static_cast<float>(K_FF)), lookahead_(static_cast<float>(LOOKAHEAD)),
      V_CRUISE_(v_cruise) {
    R_ = Eigen::Matrix2f::Zero();
    R_ << R_DELTA, 0.0f,  // 1.0
          0.0f, R_A;      // 1.0

    max_delta_ = DELTA_MAX; // 0.5 rad
    min_delta_ = -DELTA_MAX;
    max_a_ = A_MAX;        // 2.0 m/s²
    min_a_ = -A_MAX;
    R_delta_rate_ = 1.0f;  // Override R_DELTA_RATE = 0.25 (10.0 too high)
    R_a_rate_ = 0.1f;      // Low penalty for accel rate

    state_ = Eigen::Vector3f::Zero(); // [ey, yaw, v]
    delta_ = 0.0f;
    a_ = 0.0f;
    delta_prev_ = 0.0f;
    a_prev_ = 0.0f;

    V_ref_ = V_CRUISE_; // Initialize to cruise speed
}

float MPCController::estimateCurvature(float yaw) const {
    float dynamic_lookahead = lookahead_ + (state_(2) * 1.55); // state_(2) = v
    return -10.0 * yaw / dynamic_lookahead; // 1/m
}

void MPCController::adaptative(float yaw, float v) {
    static bool in_curve = false;
    float yaw_abs = std::abs(yaw);

    // Hysteresis for curve detection
    if (!in_curve && yaw_abs > YAW_HYST_HIGH) {
        in_curve = true;
    } else if (in_curve && yaw_abs < YAW_HYST_LOW) {
        in_curve = false;
    }

    // Determine target weights and V_ref
    float qEyTarget = qEyStraight;    // 1.5
    float qYawTarget = qYawStraight;  // 0.075
    float vTarget = V_CRUISE_;        // Baseline cruise speed
    float qVTarget = Q_V;             // Base 1.0
    float rATarget = R_A;             // Base 1.0

    if (in_curve) {
        if (yaw_abs > YAW_STEEP) {
            qEyTarget = qEyCurve * 1.2f;    // 12.0
            qYawTarget = qYawCurve * 1.4f;  // 0.7
            vTarget = V_CRUISE_ * 0.4f;     // Tight curve
        } else {
            qEyTarget = qEyCurve;           // 10.0
            qYawTarget = qYawCurve;         // 0.5
            vTarget = V_CRUISE_ * 0.6f;     // Moderate curve
        }
        // Further slow if current v is high
        if (v > V_CRUISE_ * 0.8f) vTarget *= 0.9f;
    }

    // Speed-dependent scaling
    float speed_factor = std::clamp(1.0f + 0.1f * (v - V_MIN), speed_factor_min, speed_factor_max);
    qEyTarget *= speed_factor;   // Tighter at high v
    qYawTarget *= speed_factor;
    qVTarget *= speed_factor;    // Enforce V_ref_ at high v
    rATarget *= (1.0f + 0.2f * v); // Smoother accel at high v

    // Low-pass filter
    qEyFilt_ = (1.0f - alpha) * qEyFilt_ + alpha * qEyTarget;
    qYawFilt_ = (1.0f - alpha) * qYawFilt_ + alpha * qYawTarget;
    qVFilt_ = (1.0f - alpha) * qVFilt_ + alpha * qVTarget;
    V_ref_ = (1.0f - alpha) * V_ref_ + alpha * vTarget;

    // Update Q and Qf
    Q_ << qEyFilt_, 0.0f, 0.0f,
          0.0f, qYawFilt_, 0.0f,
          0.0f, 0.0f, qVFilt_;
    Qf_ << qEyFilt_ * 5.0f, 0.0f, 0.0f,
           0.0f, qYawFilt_ * 5.0f, 0.0f,
           0.0f, 0.0f, qVFilt_ * 5.0f;
}

void MPCController::update(float ey, float yaw, float v) {
    // Clip velocity
    if (v < V_MIN || v > V_MAX) {
        v = V_MIN;
        std::cout << "[" << __func__ << "] Warning: Invalid v, clipped to V_MIN=" << V_MIN << std::endl;
    }
    state_ << ey, yaw, v; // Set [ey, yaw, v]

    adaptative(yaw, state_(2)); // Use state_(2) for v

    // Feed-forward
    float delta_max = max_delta_;
    float curvature = estimateCurvature(yaw);
    float delta_ff = k_ff_ * (state_(2) * state_(2) / L_) * curvature;
    static float post_curve_timer = 0.0f;
    if (std::abs(curvature) > 1.0 / 0.8) {
        delta_ff *= 1.15;
        post_curve_timer = 0.5f;
    } else if (post_curve_timer > 0) {
        delta_ff *= 1.1;
        post_curve_timer -= dt_;
    }
    delta_ff = std::clamp(delta_ff, -max_delta_, max_delta_);

    // Linearized model
    Eigen::Matrix3f A;
    Eigen::MatrixXf B(3, 2);
    A << 1.0f, dt_ * state_(2), dt_ * yaw, // ey_dot ≈ v * yaw
         0.0f, 1.0f, dt_ * (delta_ / L_),  // yaw_dot ≈ v/L * delta
         0.0f, 0.0f, 1.0f;                 // v_dot = a
    B << 0.0f, 0.0f,
         dt_ * (state_(2) / L_), 0.0f,
         0.0f, dt_;

    // MPC matrices
    Eigen::MatrixXf Ad(3 * N_, 3); Ad.setZero();
    Eigen::MatrixXf Bd(3 * N_, 2 * N_); Bd.setZero();
    Eigen::MatrixXf Qd(3 * N_, 3 * N_); Qd.setZero();
    Eigen::MatrixXf Rd = Eigen::MatrixXf::Zero(2 * N_, 2 * N_);
    for (int i = 0; i < N_; ++i) {
        Rd.block(2 * i, 2 * i, 2, 2) = R_; // Use adapted R_
    }

    for (int i = 0; i < N_; ++i) {
        Eigen::Matrix3f Apow = Eigen::Matrix3f::Identity();
        for (int j = 0; j < i; ++j) Apow = A * Apow;
        Ad.block(3 * i, 0, 3, 3) = Apow;

        for (int j = 0; j <= i; ++j) {
            Eigen::Matrix3f Apow2 = Eigen::Matrix3f::Identity();
            for (int k = 0; k < i - j; ++k) Apow2 = A * Apow2;
            Bd.block(3 * i, 2 * j, 3, 2) = Apow2 * B;
        }

        Qd.block(3 * i, 3 * i, 3, 3) = (i == N_-1) ? Qf_ : Q_;
    }

    // Reference state
    Eigen::VectorXf r(3 * N_);
    for (int i = 0; i < N_; ++i) {
        r.segment(3 * i, 3) << 0.0f, 0.0f, V_ref_;
    }

    // QP
    Eigen::MatrixXf H = 2.0f * (Bd.transpose() * Qd * Bd + Rd);
    Eigen::VectorXf f = 2.0f * Bd.transpose() * Qd * (Ad * state_ - r);

    // Rate penalties
    for (int i = 0; i < N_ - 1; ++i) {
        H(2*i, 2*i) += 2.0f * R_delta_rate_;
        H(2*(i+1), 2*i) -= 2.0f * R_delta_rate_;
        H(2*i, 2*(i+1)) -= 2.0f * R_delta_rate_;
        H(2*(i+1), 2*(i+1)) += 2.0f * R_delta_rate_;

        H(2*i+1, 2*i+1) += 2.0f * R_a_rate_;
        H(2*(i+1)+1, 2*i+1) -= 2.0f * R_a_rate_;
        H(2*i+1, 2*(i+1)+1) -= 2.0f * R_a_rate_;
        H(2*(i+1)+1, 2*(i+1)+1) += 2.0f * R_a_rate_;
    }
    f(0) += 2.0f * R_delta_rate_ * (-delta_prev_);
    f(1) += 2.0f * R_a_rate_ * (-a_prev_);

    // Projected gradient descent
    Eigen::VectorXf u = Eigen::VectorXf::Zero(2 * N_);
    Eigen::VectorXf m = Eigen::VectorXf::Zero(2 * N_);
    const float alpha = 0.005f;
    const float beta = 0.95f;

    for (int iter = 0; iter < MPC_ITER; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        u -= alpha * m;

        // Clamp controls
        for (int i = 0; i < N_; ++i) {
            u(2*i) = std::clamp(u(2*i), min_delta_, max_delta_);
            u(2*i + 1) = std::clamp(u(2*i + 1), min_a_, max_a_);
        }
        // Rate constraints
        for (int i = 1; i < N_; ++i) {
            float delta_rate = u(2*i) - u(2*(i-1));
            if (delta_rate > DELTA_RATE_MAX) u(2*i) = u(2*(i-1)) + DELTA_RATE_MAX;
            if (delta_rate < -DELTA_RATE_MAX) u(2*i) = u(2*(i-1)) - DELTA_RATE_MAX;

            float a_rate = u(2*i + 1) - u(2*(i-1) + 1);
            if (a_rate > 0.5f) u(2*i + 1) = u(2*(i-1) + 1) + 0.5f;
            if (a_rate < -0.5f) u(2*i + 1) = u(2*(i-1) + 1) - 0.5f;
        }

        if (grad.norm() < 1e-4f) break;
    }

    delta_prev_ = delta_;
    a_prev_ = a_;
    delta_ = std::clamp(delta_ff + u(0), min_delta_, max_delta_);
    a_ = u(1);
}

float MPCController::getSteeringAngle() const {
    return delta_;
}

float MPCController::getAcceleration() const {
    return a_;
}