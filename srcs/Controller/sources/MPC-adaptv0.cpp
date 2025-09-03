// Version: v11 (2025-08-28) FEED-FORWARD CONTROL ADDED
#include "MPC.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

/// @brief 	Constructor for MPCController
/// @param wheelbase	lenghth between front and rear axles (m)
/// @param dt			time step (s)
/// @param horizon		prediction horizon (number of steps)
MPCController::MPCController(float wheelbase, float dt, int horizon)
    : L_(wheelbase),
	dt_(dt),
	N_(horizon),
	k_ff_(static_cast<float>(K_FF)),
	lookahead_(static_cast<float>(LOOKAHEAD))
{
	// Q_ << Q_EY, 0.0f,
	// 	  0.0f, Q_YAW; // ey: 20, yaw: 5
    // Qf_ << QF_EY, 0.0f,
	// 	   0.0f, QF_YAW; // ey: 100, yaw: 25

	R_ = R; // Balanced control effort
    max_delta_ = DELTA_MAX; // 0.5 rad
	min_delta_ = -DELTA_MAX; // -0.5 rad
    R_delta_rate_ = R_DELTA_RATE; // 10.0

	k_delta_ = 0.25f; // Unused (speed-dependent delta)

    state_ = Eigen::Vector2f::Zero(); // [ey, yaw]
    v_ = 0.0f; // Initial speed
	delta_ = 0.0f; // Initial steering
    delta_prev_ = 0.0f; // For rate penalty

	// unused for now
	a_ = 0.0f; // Initial acceleration
}

float MPCController::estimateCurvature(float yaw) const
{
    /* Aproximação grosseira: curvature ≈ yaw / distância-olho.
       Ajuste conforme sua geometria ou use visão.            */
    float dynamic_lookahead = lookahead_ + (v_ * 1.55);  // (m)
    return -10.0 * yaw / dynamic_lookahead;                // (1/m)
}

void MPCController::adaptative(float yaw, float v) {
	// Unused for now
	(void)v;

	float qYaw = Q_YAW;
	if (std::abs(yaw) < 0.3f) {
		qYaw = Q_YAW * 0.8f;  // Reduce weight for small yaws
	} else if (std::abs(yaw) > 0.2f && std::abs(yaw) < 0.4f) {
		qYaw = Q_YAW;  // Increase weight for large yaws
	} else if (std::abs(yaw) > 0.4f && std::abs(yaw) < 0.5f){
		qYaw = Q_YAW * 1.2f;  // Increase weight for large yawselse {
	} else {
		qYaw = 10.0f;  // Normal weight
	}

	// Example: Adjust Q_EY based on speed (higher speed -> lower weight)
	Q_ << Q_EY, 0.0f,
		  0.0f, qYaw; // ey: 20, yaw: 5 0.2 is the smallest yaw expected
    Qf_ << Q_EY * 5.0, 0.0f,
		   0.0f, qYaw * 5.0; // ey: 100, yaw: 25

}

void MPCController::update(float ey, float yaw, float v) {
    // Clip invalid velocity
    if (v < 0.0f || v > V_MAX) {
        v = V_MIN;
        std::cout << "[" << __func__ << "] Warning: Invalid v clipped to V_MIN=" << V_MIN << std::endl;
    }

    state_ << ey, yaw;
    v_ = std::max(0.1f, v);

	adaptative(yaw, v_);



    float delta_max = max_delta_; // 0.5 rad

    float curvature = estimateCurvature(yaw);  // 1/m
	float delta_ff = k_ff_ * (v_ * v_ / L_) * curvature;
	static float post_curve_timer = 0.0f;  // Persist across calls
	if (std::abs(curvature) > 1.0 / 0.8) {
		delta_ff *= 1.15;
		post_curve_timer = 0.5f;  // 0.5s post-curve boost (adjust)
	} else if (post_curve_timer > 0) {
		delta_ff *= 1.1;  // Mild boost during recovery
		post_curve_timer -= dt_;
	}
	delta_ff = std::clamp(delta_ff, -max_delta_, max_delta_);



	// Linearized model for QP: x(k+1) = A * x(k) + B * u(k)
	Eigen::Matrix2f A;
	Eigen::Vector2f B;
	A << 1.0f, dt_ * v_, 0.0f, 1.0f;
	B << dt_ * v_, dt_ * (v_ / L_);

	// MPC matrices for horizon pilling up
    Eigen::MatrixXf Ad(2 * N_, 2); Ad.setZero();
    Eigen::MatrixXf Bd(2 * N_, N_); Bd.setZero();
    Eigen::MatrixXf Qd(2 * N_, 2 * N_); Qd.setZero();
    Eigen::MatrixXf Rd = Eigen::MatrixXf::Identity(N_, N_) * R_;

    for (int i = 0; i < N_; ++i) {
        Eigen::Matrix2f Apow = Eigen::Matrix2f::Identity();
        for (int j = 0; j < i; ++j) Apow = A * Apow;
        Ad.block(2 * i, 0, 2, 2) = Apow;

        for (int j = 0; j <= i; ++j) {
            Eigen::Matrix2f Apow2 = Eigen::Matrix2f::Identity();
            for (int k = 0; k < i - j; ++k) Apow2 = A * Apow2;
            Bd.block(2 * i, j, 2, 1) = Apow2 * B;
        }

        Qd.block(2 * i, 2 * i, 2, 2) = (i == N_-1) ? Qf_ : Q_;
    }

	// Reference state (straight lane for now)
    Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N_);

    /* ---------- QP: H u + f ---------- */
    Eigen::MatrixXf H = 2.0f * (Bd.transpose() * Qd * Bd + Rd);
    Eigen::VectorXf  f = 2.0f * Bd.transpose() * Qd * (Ad * state_ - r);

    /* delta variation penalty */
    for (int i = 0; i < N_ - 1; ++i) {
        H(i,i)                 += 2.0f * R_delta_rate_;
        H(i+1,i)               -= 2.0f * R_delta_rate_;
        H(i,i+1)               -= 2.0f * R_delta_rate_;
        H(i+1,i+1)             += 2.0f * R_delta_rate_;
    }
    f(0) += 2.0f * R_delta_rate_ * (-delta_prev_);

    /* ---------- projected-gradient descent ---------- */
    Eigen::VectorXf u  = Eigen::VectorXf::Zero(N_);
    Eigen::VectorXf m  = Eigen::VectorXf::Zero(N_);
    const float alpha  = 0.005f;    // learning-rate
    const float beta   = 0.95f;     // momentum

    for (int iter = 0; iter < MPC_ITER; ++iter) {
        Eigen::VectorXf grad = H * u + f;
        m = beta * m + (1.0f - beta) * grad;
        u -= alpha * m;

        /* projeção nos limites físicos */
        u = u.cwiseMax(-delta_max).cwiseMin(delta_max);
        for (int i = 1; i < N_; ++i) {
            float rate = u(i) - u(i-1);
            if (rate >  DELTA_RATE_MAX) u(i) = u(i-1) + DELTA_RATE_MAX;
            if (rate < -DELTA_RATE_MAX) u(i) = u(i-1) - DELTA_RATE_MAX;
        }

        if (grad.norm() < 1e-4f) break;
    }

    delta_prev_ = delta_;
	// std::cout << v_ << "\t" << ey << "\t" << yaw << "\t" << curvature << "\t" << delta_ff << "\t" << u(0) << "\t" << Q_(0,0) << "\t" << Q_(1,1) << std::endl;
    delta_ = std::clamp(delta_ff + u(0), -delta_max, delta_max);

}

float MPCController::getSteeringAngle() const {
    return delta_; // Confirmed correct sign
}

float MPCController::getAcceleration() const {
    return 0.0f; // No acceleration control
}
// End of MPC.cpp