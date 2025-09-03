// Version: v9 (2025-08-28)
#ifndef MPC_HPP
#define MPC_HPP

#include <Eigen/Dense>
#include "Configs.hpp"

class MPCController {
public:
	MPCController(float wheelbase, float dt, int horizon, float v_cruise);
	// Update state and compute optimal control
	void update(float ey, float yaw, float v);
	// Control outputs
	float getSteeringAngle() const;
	float getAcceleration() const;

	private:
	// fixed parameters
	float L_; // Wheelbase (m)
	float dt_; // Time step (s)
	int N_; // Prediction horizon

	Eigen::Matrix2f	Q_; // State cost matrix
	Eigen::Matrix2f	Qf_; // Terminal cost matrix
	float 			R_; // Control cost
	float 			max_delta_; // Max physical steering angle (rad)
	float 			k_delta_; // Speed-dependent delta constant (rad·m/s)
	float 			min_delta_; // Minimum delta limit (rad)

	// dynamic states
	Eigen::Vector2f	state_; // [ey, yaw]
	float 			v_; // Current speed (m/s)
	float 			delta_; // Steering angle (rad)
	float 			a_; // Acceleration (m/s^2)
	float 			delta_prev_; // Previous delta for rate penalty
	float 			R_delta_rate_; // Penalty for delta rate change

	// feed-forward control
	float			k_ff_; // Feed-forward gain
	float			lookahead_;
	float 			estimateCurvature(float yaw) const;

	// adaptative controller
	void adaptative(float yaw, float v);
	// Add these to your class private section
	float qEyFilt_			  = Q_EY;	// Smoothed Q_EY
	float qYawFilt_			 = Q_YAW;  // Smoothed Q_YAW

	// Parameters for adaptation
	const float qEyStraight		= Q_EY * 0.3f;	// Low weight for straight
	const float qEyCurve		= Q_EY * 2.0f;	// High weight for sharper curve
	const float qYawStraight	= Q_YAW * 0.3f;
	const float qYawCurve		= Q_YAW * 2.0f;

	const float alpha = 0.15f;	// Interpolation factor, 0.10-0.20 for filtering

	const float YAW_HYST_LOW	= 0.08f;
	const float YAW_HYST_HIGH   = 0.15f;
	const float YAW_STEEP		= 0.35f;

};

#endif // MPC_HPP