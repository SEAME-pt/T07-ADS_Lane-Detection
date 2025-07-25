// Version: v8 (2025-07-24)
#include "MPC.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase), dt_(dt), N_(horizon) {
    Q_ << Q_EY, 0.0f, 0.0f, Q_PSI_ERR; // ey: 50, yaw: 50
    R_ = R_DELTA; // 1.0
    Qf_ << 2.0f * Q_EY, 0.0f, 0.0f, 2.0f * Q_PSI_ERR; // ey: 100, yaw: 100
    max_delta_ = DELTA_MAX; // 0.523 rad
    k_delta_ = 0.1f; // Kept from v4 for curves
    min_delta_ = 0.05f; // Minimum delta limit
    state_ = Eigen::Vector2f::Zero(); // [ey, yaw]
    delta_ = 0.0f;
    delta_prev_ = 0.0f;
}

void MPCController::update(float ey, float yaw, float v) {
    // Deadband for small errors (from v5)
    float ey_adj = (std::abs(ey) < 0.02f) ? 0.0f : ey;
    float yaw_adj = (std::abs(yaw) < 0.02f) ? 0.0f : yaw;
    state_ << ey_adj, yaw_adj; // ey positive left, yaw positive heading left
    v_ = std::max(0.1f, v); // Avoid division by zero

    // Speed-dependent delta limit with hard cap
    float delta_max = std::min({max_delta_, k_delta_ / v_, 0.095f}); // Cap at 0.095 rad
    delta_max = std::max(min_delta_, delta_max);

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

    // Delta rate penalty, reduced for curves
    float R_delta_rate = (std::abs(ey) < 0.05f && std::abs(yaw) < 0.05f) ? 50.0f : 20.0f;
    for (int i = 0; i < N_ - 1; ++i) {
        H(i, i) += 2.0f * R_delta_rate;
        H(i + 1, i) -= 2.0f * R_delta_rate;
        H(i, i + 1) -= 2.0f * R_delta_rate;
        H(i + 1, i + 1) += 2.0f * R_delta_rate;
    }
    f(0) += 2.0f * R_delta_rate * (-delta_prev_);

    // Constraints: |delta| <= delta_max, |delta - delta_prev| <= DELTA_RATE_MAX
    Eigen::MatrixXf A_constr(2 * N_, N_);
    Eigen::VectorXf lb_constr(2 * N_);
    Eigen::VectorXf ub_constr(2 * N_);
    A_constr.setZero();
    for (int i = 0; i < N_; ++i) {
        A_constr(i, i) = 1.0f; // delta constraints
        A_constr(N_ + i, i) = 1.0f; // delta rate constraints
        if (i > 0) {
            A_constr(N_ + i, i - 1) = -1.0f;
        }
    }
    lb_constr.head(N_).setConstant(-delta_max);
    ub_constr.head(N_).setConstant(delta_max);
    lb_constr.tail(N_).setConstant(-DELTA_RATE_MAX);
    ub_constr.tail(N_).setConstant(DELTA_RATE_MAX);
    if (N_ > 0) {
        A_constr(N_, 0) = 1.0f;
        lb_constr(N_) = -delta_max - delta_prev_;
        ub_constr(N_) = delta_max - delta_prev_;
    }

    // Projected gradient descent with adaptive step size
    Eigen::VectorXf u = Eigen::VectorXf::Zero(N_);
    Eigen::VectorXf u_prev = u;
    float alpha = 0.01f;
    float beta = 0.9f;
    Eigen::VectorXf m = Eigen::VectorXf::Zero(N_);
    for (int iter = 0; iter < 200; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        Eigen::VectorXf u_new = u - alpha * m;
        for (int i = 0; i < N_; ++i) {
            u_new(i) = std::max(lb_constr(i), std::min(ub_constr(i), u_new(i)));
            if (i > 0) {
                u_new(i) = std::max(lb_constr(N_ + i) + u_new(i - 1),
                                   std::min(ub_constr(N_ + i) + u_new(i - 1), u_new(i)));
            } else {
                u_new(i) = std::max(-delta_max - delta_prev_ + delta_prev_,
                                   std::min(delta_max - delta_prev_ + delta_prev_, u_new(i)));
            }
        }
        float cost_old = (0.5f * u.transpose() * H * u + f.transpose() * u).value();
        float cost_new = (0.5f * u_new.transpose() * H * u_new + f.transpose() * u_new).value();
        if (cost_new > cost_old && alpha > 0.001f) {
            alpha *= 0.5f;
        } else {
            u = u_new;
            if ((u - u_prev).norm() < 1e-4f) break;
            u_prev = u;
            alpha = std::min(alpha * 1.2f, 0.01f);
        }
    }

    delta_prev_ = delta_;
    delta_ = u(0);
    std::cout << "[" << __func__ << "] :"
              << "\n\tMPC (L, DT, N) : (" << L_ << ", " << dt_ << ", " << N_ << ")"
              << "\n\tey        : " << ey << " m, yaw: " << yaw << " rad"
              << "\n\tv         : " << v << " m/s"
              << "\n\tdelta     : " << delta_ << " rad"
              << "\n\tdelta_max : " << delta_max << " rad"
              << std::endl;
}

float MPCController::getSteeringAngle() const {
    return delta_;
}

float MPCController::getAcceleration() const {
    return 0.0f;
}