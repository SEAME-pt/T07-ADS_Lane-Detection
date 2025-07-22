#include "MPCController.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase), dt_(dt), N_(horizon) {
    Q_ = Eigen::Matrix2f::Identity() * 20.0f; // Weight on ey, yaw
    R_ = 2.0f; // Weight on delta
    Qf_ = Eigen::Matrix2f::Identity() * 40.0f; // Terminal weight
    max_delta_ = 0.5f; // Max physical steering angle (rad)
    k_delta_ = 0.15f; // Speed-dependent delta constant (rad·m/s)
    min_delta_ = 0.03f; // Minimum delta limit (rad)
    state_ = Eigen::Vector2f::Zero(); // [ey, yaw]
    delta_ = 0.0f;
}

void MPCController::update(float ey, float yaw, float v) {
    state_ << ey, yaw; // ey positive left, yaw positive heading left
    v_ = std::max(0.1f, v); // Avoid division by zero, min speed 0.1 m/s

    // Compute speed-dependent delta limit
    float delta_max = std::min(max_delta_, k_delta_ / v_);
    delta_max = std::max(min_delta_, delta_max);

    // Discretized bicycle model: x(k+1) = A * x(k) + B * u(k)
    Eigen::Matrix2f A;
    Eigen::Vector2f B;
    float yaw_rate = (v_ / L_) * std::tan(delta_); // Linearize around current delta
    A << 1.0f, dt_ * v_, 0.0f, 1.0f;
    B << dt_ * v_ * std::cos(yaw + delta_), dt_ * (v_ / L_) / (std::cos(delta_) * std::cos(delta_));

    // MPC matrices
    Eigen::MatrixXf Ad(2 * N_, 2); // Augmented state transition
    Eigen::MatrixXf Bd(2 * N_, N_); // Augmented control input
    Eigen::MatrixXf Qd(2 * N_, 2 * N_); // Cost matrix for states
    Eigen::MatrixXf Rd(N_, N_); // Cost matrix for controls
    Ad.setZero();
    Bd.setZero();
    Qd.setZero();
    Rd.setIdentity() * R_;

    // Build augmented matrices
    for (int i = 0; i < N_; ++i) {
        // State transition: A^i
        Eigen::Matrix2f Apow = Eigen::Matrix2f::Identity();
        for (int j = 0; j < i; ++j) {
            Apow = A * Apow;
        }
        Ad.block(2 * i, 0, 2, 2) = Apow;

        // Control input: sum(A^j * B)
        for (int j = 0; j <= i; ++j) {
            Apow = Eigen::Matrix2f::Identity();
            for (int k = 0; k < i - j; ++k) {
                Apow = A * Apow;
            }
            Bd.block(2 * i, j, 2, 1) = Apow * B;
        }

        // Cost matrix
        Qd.block(2 * i, 2 * i, 2, 2) = (i == N_ - 1) ? Qf_ : Q_;
    }

    // Reference state (track ey = 0, yaw = 0)
    Eigen::VectorXf r(2 * N_);
    r.setZero();

    // Compute H and f for QP: min (0.5 * u^T * H * u + f^T * u)
    Eigen::MatrixXf H = 2.0f * (Bd.transpose() * Qd * Bd + Rd);
    Eigen::VectorXf f = 2.0f * Bd.transpose() * Qd * (Ad * state_ - r);

    // Constraints: |delta| <= delta_max
    Eigen::MatrixXf A_constr(N_, N_);
    Eigen::VectorXf lb_constr(N_);
    Eigen::VectorXf ub_constr(N_);
    A_constr.setIdentity();
    lb_constr.setConstant(-delta_max);
    ub_constr.setConstant(delta_max);

    // Simple QP solver (gradient descent for simplicity, replace with OSQP for production)
    Eigen::VectorXf u = Eigen::VectorXf::Zero(N_);
    float alpha = 0.005f; // Smaller step size for stability
    for (int iter = 0; iter < 150; ++iter) { // More iterations for convergence
        Eigen::VectorXf grad = H * u + f;
        u -= alpha * grad;
        // Project onto constraints
        for (int i = 0; i < N_; ++i) {
            u(i) = std::max(lb_constr(i), std::min(ub_constr(i), u(i)));
        }
    }

    delta_ = u(0); // Apply first control input
    std::cout << "[" << __func__ << "] ey: " << ey << " m, yaw: " << yaw
              << " rad, v: " << v << " m/s, delta: " << delta_ << " rad"
              << ", delta_max: " << delta_max << std::endl;
}

float MPCController::getSteeringAngle() const {
    return delta_;
}

float MPCController::getAcceleration() const {
    return 0.0f; // Not controlling acceleration
}