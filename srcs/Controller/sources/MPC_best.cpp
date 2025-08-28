// Version: v10 (2025-08-28)
#include "MPC.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase), dt_(dt), N_(horizon) {
    R_ = R; // Balanced control effort
    Q_ << Q_EY, 0.0f, 0.0f, Q_YAW; // ey: 20, yaw: 5
    Qf_ << QF_EY, 0.0f, 0.0f, QF_YAW; // ey: 100, yaw: 25
    max_delta_ = DELTA_MAX; // 0.5 rad
    k_delta_ = 0.25f; // Unused (speed-dependent delta)
    min_delta_ = 0.05f; // Minimum delta limit
    state_ = Eigen::Vector2f::Zero(); // [ey, yaw]
    delta_ = 0.0f; // Initial steering
	a_ = 0.0f; // Initial acceleration
    delta_prev_ = 0.0f; // For rate penalty
    R_delta_rate_ = R_DELTA_RATE; // 10.0
}

void MPCController::update(float ey, float yaw, float v) {
    // Clip invalid velocity
    if (v < 0.0f || v > V_MAX) {
        v = V_MIN;
        std::cout << "[" << __func__ << "] Warning: Invalid v clipped to V_MIN=" << V_MIN << std::endl;
    }

    state_ << ey, yaw;
    v_ = std::max(0.1f, v);

    float delta_max = max_delta_; // 0.5 rad

    // Nonlinear bicycle model for prediction
    Eigen::VectorXf u_pred(N_);
    u_pred.setZero();
    Eigen::VectorXf states(2 * N_);
    states.segment(0, 2) = state_;
    for (int i = 0; i < N_ - 1; ++i) {
        float curr_yaw = states(2 * i + 1);
        float curr_delta = i == 0 ? delta_ : u_pred(i);
        states(2 * (i + 1)) = states(2 * i) + dt_ * v_ * std::sin(curr_yaw + curr_delta);
        states(2 * (i + 1) + 1) = states(2 * i + 1) + dt_ * (v_ / L_) * std::tan(curr_delta);
    }

    // Linearized model for QP: x(k+1) = A * x(k) + B * u(k)
    Eigen::Matrix2f A;
    Eigen::Vector2f B;
    A << 1.0f, dt_ * v_, 0.0f, 1.0f;
    B << dt_ * v_, dt_ * (v_ / L_);

    // MPC matrices
    Eigen::MatrixXf Ad(2 * N_, 2);
    Eigen::MatrixXf Bd(2 * N_, N_);
    Eigen::MatrixXf Qd(2 * N_, 2 * N_);
    Eigen::MatrixXf Rd(N_, N_);
    Ad.setZero();
    Bd.setZero();
    Qd.setZero();
    Rd.setIdentity() * R_;

    for (int i = 0; i < N_; ++i) {
        Eigen::Matrix2f Apow = Eigen::Matrix2f::Identity();
        for (int j = 0; j < i; ++j) {
            Apow = A * Apow;
        }
        Ad.block(2 * i, 0, 2, 2) = Apow;

        for (int j = 0; j <= i; ++j) {
            Apow = Eigen::Matrix2f::Identity();
            for (int k = 0; k < i - j; ++k) {
                Apow = A * Apow;
            }
            Bd.block(2 * i, j, 2, 1) = Apow * B;
        }
        Qd.block(2 * i, 2 * i, 2, 2) = (i == N_ - 1) ? Qf_ : Q_;
    }

    // Reference state (straight lane for now)
    Eigen::VectorXf r(2 * N_);
    r.setZero();

    // QP: min (0.5 * u^T * H * u + f^T * u)
    Eigen::MatrixXf H = 2.0f * (Bd.transpose() * Qd * Bd + Rd);
    Eigen::VectorXf f = 2.0f * Bd.transpose() * Qd * (Ad * state_ - r);

    // Add delta rate penalty
    float R_delta_rate = R_delta_rate_; // 10.0
    for (int i = 0; i < N_ - 1; ++i) {
        H(i, i) += 2.0f * R_delta_rate;
        H(i + 1, i) -= 2.0f * R_delta_rate;
        H(i, i + 1) -= 2.0f * R_delta_rate;
        H(i + 1, i + 1) += 2.0f * R_delta_rate;
    }
    f(0) += 2.0f * R_delta_rate * (-delta_prev_);

    // Constraints: delta and rate bounds
    Eigen::MatrixXf A_constr(2 * N_, N_);
    Eigen::VectorXf lb_constr(2 * N_);
    Eigen::VectorXf ub_constr(2 * N_);
    A_constr.setZero();
    A_constr.block(0, 0, N_, N_) = Eigen::MatrixXf::Identity(N_, N_);
    lb_constr.segment(0, N_) = Eigen::VectorXf::Constant(N_, -delta_max);
    ub_constr.segment(0, N_) = Eigen::VectorXf::Constant(N_, delta_max);
    for (int i = 1; i < N_; ++i) {
        A_constr(N_ + i - 1, i) = 1.0f;
        A_constr(N_ + i - 1, i - 1) = -1.0f;
        lb_constr(N_ + i - 1) = -DELTA_RATE_MAX;
        ub_constr(N_ + i - 1) = DELTA_RATE_MAX;
    }

    // Projected gradient descent with momentum
    Eigen::VectorXf u = Eigen::VectorXf::Zero(N_);
    Eigen::VectorXf u_prev = u;
    float alpha = 0.01f; // Balanced for convergence
    float beta = 0.9f;
    Eigen::VectorXf m = Eigen::VectorXf::Zero(N_);
    for (int iter = 0; iter < MPC_ITER; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        u -= alpha * m;
        for (int i = 0; i < N_; ++i) {
            u(i) = std::max(-delta_max, std::min(delta_max, u(i)));
        }
        for (int i = 1; i < N_; ++i) {
            float rate = u(i) - u(i - 1);
            if (rate > DELTA_RATE_MAX) u(i) = u(i - 1) + DELTA_RATE_MAX;
            if (rate < -DELTA_RATE_MAX) u(i) = u(i - 1) - DELTA_RATE_MAX;
        }
        if ((u - u_prev).norm() < 1e-4f) {
            break;
        }
        u_prev = u;
    }

    delta_prev_ = delta_;
    delta_ = u(0);
}

float MPCController::getSteeringAngle() const {
    return delta_; // Confirmed correct sign
}

float MPCController::getAcceleration() const {
    return 0.0f; // No acceleration control
}
// End of MPC.cpp