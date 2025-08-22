#include "MPC.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase), dt_(dt), N_(horizon) {
    R_ = 10.0f; // Balanced control effort
    Q_ << Q_EY, 10.0f, 10.0f, Q_YAW; // ey:40, yaw:15, off-diagonal for coupled errors
    Qf_ << QF_EY, 50.0f, 50.0f, QF_YAW; // ey:200, yaw:75, off-diagonal
    max_delta_ = DELTA_MAX; // 0.5 rad
    k_delta_ = 0.5f;
    min_delta_ = 0.05f;
    state_ = Eigen::Vector2f::Zero();
    delta_ = 0.0f;
    delta_prev_ = 0.0f;
    R_delta_rate_ = R_DELTA_RATE; // 15.0
}

void MPCController::update(float ey, float yaw, float v) {
    static int log_count = 0;
    static std::ofstream log_file("mpc_logs.txt", std::ios::app);
    static auto start_time = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    double timestamp = std::chrono::duration<double>(now - start_time).count();

    if (v < 0.0f || v > V_MAX) {
        v = V_REF;
        std::cout << "[" << __func__ << "] Warning: Invalid v clipped to V_REF=" << V_REF << std::endl;
    }

    state_ << ey, yaw;
    if (v < 0.1f) {
        delta_ = 0.0f;
        std::cout << "[" << __func__ << "] Warning: Speed is too LOW => LKAS : OFF" << std::endl;
        if (log_count < 200 && log_file.is_open()) {
            log_file << std::fixed << std::setprecision(6)
                     << "Entry " << log_count + 1 << ": "
                     << "t=" << timestamp << ", ey=" << ey << ", yaw=" << yaw
                     << ", v=" << v << ", delta=" << delta_ << ", delta_max=0.0, iter=0"
                     << std::endl;
            log_count++;
            if (log_count == 200) {
                log_file << "Reached 200 logs. Stopping file logging." << std::endl;
                log_file.close();
            }
        }
        return;
    }
    v_ = std::max(0.1f, v);

    float delta_max = max_delta_;

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

    // Reference state (ey = 0, yaw = 0)
    Eigen::VectorXf r(2 * N_);
    r.setZero();

    // QP: min (0.5 * u^T * H * u + f^T * u)
    Eigen::MatrixXf H = 2.0f * (Bd.transpose() * Qd * Bd + Rd);
    Eigen::VectorXf f = 2.0f * Bd.transpose() * Qd * (Ad * state_ - r);

    // Add delta rate penalty to reduce oscillations
    float R_delta_rate = R_delta_rate_; // 15.0
    for (int i = 0; i < N_ - 1; ++i) {
        H(i, i) += 2.0f * R_delta_rate;
        H(i + 1, i) -= 2.0f * R_delta_rate;
        H(i, i + 1) -= 2.0f * R_delta_rate;
        H(i + 1, i + 1) += 2.0f * R_delta_rate;
    }
    f(0) += 2.0f * R_delta_rate * (-delta_prev_);

    // Constraints: |delta| <= delta_max
    Eigen::MatrixXf A_constr(N_, N_);
    Eigen::VectorXf lb_constr(N_);
    Eigen::VectorXf ub_constr(N_);
    A_constr.setIdentity();
    lb_constr.setConstant(-delta_max);
    ub_constr.setConstant(delta_max);

    // Projected gradient descent with momentum
    Eigen::VectorXf u = Eigen::VectorXf::Zero(N_);
    Eigen::VectorXf u_prev = u;
    float alpha = 0.005f; // Lowered for stability
    float beta = 0.9f;
    Eigen::VectorXf m = Eigen::VectorXf::Zero(N_);
    int iter_used = 0;
    for (int iter = 0; iter < MPC_ITER; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        u -= alpha * m;
        for (int i = 0; i < N_; ++i) {
            u(i) = std::max(lb_constr(i), std::min(ub_constr(i), u(i)));
        }
        // Multi-pass rate projection
        for (int pass = 0; pass < 3; ++pass) {
            for (int i = 1; i < N_; ++i) {
                float rate = u(i) - u(i - 1);
                if (rate > DELTA_RATE_MAX) {
                    u(i) = u(i - 1) + DELTA_RATE_MAX;
                } else if (rate < -DELTA_RATE_MAX) {
                    u(i) = u(i - 1) - DELTA_RATE_MAX;
                }
            }
        }
        if ((u - u_prev).norm() < 1e-4f) {
            iter_used = iter + 1;
            break;
        }
        u_prev = u;
        iter_used = iter + 1;
    }

    delta_prev_ = delta_;
    delta_ = u(0);

    // Console log
    std::cout << "[" << __func__ << "] :"
				<< "\n\tMPC (L, DT, N) : " << L_ << ", " << dt_ << ", " << N_ << ")"
				<< "\n\tey        : " << ey << " m, yaw: " << yaw << " rad"
				<< "\n\tv         : " << v << " m/s"
				<< "\n\tdelta     : " << delta_ << " rad"
              	<< "\n\tdelta_max : " << delta_max << " rad"
				<< std::endl;

    // File log
    if (log_count < 200 && log_file.is_open()) {
        log_file << std::fixed << std::setprecision(6)
                 << "Entry " << log_count + 1 << ": "
                 << "t=" << timestamp << ", ey=" << ey << ", yaw=" << yaw
                 << ", v=" << v << ", delta=" << delta_ << ", delta_max=" << delta_max
                 << ", iter=" << iter_used << std::endl;
        log_count++;
        if (log_count == 200) {
            log_file << "Reached 200 logs. Stopping file logging." << std::endl;
            log_file.close();
        }
    }
}

float MPCController::getSteeringAngle() const {
    return delta_;
}

float MPCController::getAcceleration() const {
    return 0.0f;
}
// End of MPC.cpp