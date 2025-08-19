// Version: v4 (2025-07-24)
#include "MPC.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase), dt_(dt), N_(horizon) {
		R_ = 10.0f; // Control cost for smoothness
		Q_ << Q_EY, 0.0f, 0.0f, Q_YAW; // ey: 50, yaw: 50
		Qf_ << QF_EY, 0.0f, 0.0f,QF_YAW; // ey: 100, yaw: 100		// Q_ = Eigen::Matrix2f::Identity() * 100.0f; // High weight on ey, yaw
		// Qf_ = Eigen::Matrix2f::Identity() * 200.0f; // Terminal weight
		max_delta_ = DELTA_MAX; // Max physical steering angle (rad)
		k_delta_ = 0.5f; // Speed-dependent delta constant (rad·m/s)
		min_delta_ = 0.05f; // Minimum delta limit (rad)
		state_ = Eigen::Vector2f::Zero(); // [ey, yaw]
		delta_ = 0.0f; // Initial steering angle
		delta_prev_ = 0.0f; // For delta rate penalty
		R_delta_rate_ = 5.0f;


}

void MPCController::update(float ey, float yaw, float v) {
    state_ << ey, yaw; // ey positive left, yaw positive heading left
    if (v < 0.1f) {
		std::cout << "[" << __func__ << "] Warning: Speed is too LOW => LKAS : OFF" << std::endl;
		delta_ = 0.0f;
		return;
	}
	v_ = std::max(0.1f, v); // Avoid division by zero

    // Speed-dependent delta limit
	// float speedRatio =  std::max(0.0f, std::min(1.0f, static_cast<float>(v_ / V_MAX)));
    // float delta_max = std::min(max_delta_, static_cast<float>(k_delta_ * (1.0f - speedRatio)) );
    // delta_max = std::max(min_delta_, delta_max);
	float delta_max = max_delta_; // Cap at 0.09 rad
	// std::cout << "["<< __func__ << "] : speedRatio : " << speedRatio << std::endl;

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
    float R_delta_rate = R_DELTA_RATE; // Penalty on delta change
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
    float alpha = 0.01f;
    float beta = 0.9f;
    Eigen::VectorXf m = Eigen::VectorXf::Zero(N_);
    for (int iter = 0; iter < MPC_ITER; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        u -= alpha * m;
        for (int i = 0; i < N_; ++i) {
            u(i) = std::max(lb_constr(i), std::min(ub_constr(i), u(i)));
        }
        if ((u - u_prev).norm() < 1e-4f) break;
        u_prev = u;
    }

    delta_prev_ = delta_;
    delta_ = u(0);
    std::cout << "[" << __func__ << "] :"
				<< "\n\tMPC (L, DT, N) : " << L_ << ", " << dt_ << ", " << N_ << ")"
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