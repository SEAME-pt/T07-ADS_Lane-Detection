// Version: v4 (2025-07-24)
#ifndef MPC_HPP
#define MPC_HPP

#include <Eigen/Dense>
#include "Configs.hpp"

class MPCController {
public:
    MPCController(float wheelbase, float dt, int horizon);
    void update(float ey, float yaw, float v);
    float getSteeringAngle() const;
    float getAcceleration() const;

private:
    float L_; // Wheelbase (m)
    float dt_; // Time step (s)
    int N_; // Prediction horizon
    Eigen::Matrix2f Q_; // State cost matrix
    float R_; // Control cost
    Eigen::Matrix2f Qf_; // Terminal cost matrix
    float max_delta_; // Max physical steering angle (rad)
    float k_delta_; // Speed-dependent delta constant (rad·m/s)
    float min_delta_; // Minimum delta limit (rad)
    Eigen::Vector2f state_; // [ey, yaw]
    float v_; // Current speed (m/s)
    float delta_; // Steering angle (rad)
    float delta_prev_; // Previous delta for rate penalty
	float R_delta_rate_; // Penalty for delta rate change
};

#endif // MPC_HPP