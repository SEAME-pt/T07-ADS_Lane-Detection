// MPC.hpp
// Version: v13 (2025-09-03)
#ifndef MPC_HPP
#define MPC_HPP

#include <Eigen/Dense>
#include "Configs.hpp"

class MPCController {
public:
    MPCController(float wheelbase, float dt, int horizon, float v_cruise = V_MAX);
    void update(float ey, float yaw, float v);
    float getSteeringAngle() const;
    float getAcceleration() const;

private:
    // Fixed parameters
    float L_; // Wheelbase (m)
    float dt_; // Time step (s)
    int N_; // Prediction horizon

    Eigen::Matrix3f Q_;  // State cost matrix [ey, yaw, v]
    Eigen::Matrix3f Qf_; // Terminal cost matrix
    Eigen::Matrix2f R_;  // Control cost matrix [delta, a]
    float max_delta_;    // Max steering angle (rad)
    float min_delta_;    // Min steering angle (rad)
    float max_a_;        // Max acceleration (m/s²)
    float min_a_;        // Min acceleration (m/s²)
    float R_delta_rate_; // Penalty for delta rate
    float R_a_rate_;     // Penalty for acceleration rate

    // Dynamic states
    Eigen::Vector3f state_; // [ey, yaw, v]
    float delta_;           // Steering angle (rad)
    float a_;               // Acceleration (m/s²)
    float delta_prev_;      // Previous delta
    float a_prev_;          // Previous a

    // Feed-forward control
    float k_ff_;
    float lookahead_;
    float estimateCurvature(float yaw) const;

    // Adaptive controller
    void adaptative(float yaw, float v);
    float qEyFilt_ = Q_EY;    // Smoothed Q_EY
    float qYawFilt_ = Q_YAW;  // Smoothed Q_YAW
    float qVFilt_ = Q_V;      // Smoothed Q_V
    float V_CRUISE_;          // Baseline cruise speed
    float V_ref_;             // Dynamic MPC reference velocity

    // Adaptation parameters
    const float qEyStraight = Q_EY * 0.3f;     // 1.5
    const float qEyCurve = Q_EY * 2.0f;        // 10.0
    const float qYawStraight = Q_YAW * 0.3f;   // 0.075
    const float qYawCurve = Q_YAW * 2.0f;      // 0.5
    const float vStraight = V_MAX;             // 2.5 m/s
    const float vCurve = V_MAX * 0.6f;         // 1.5 m/s
    const float vSteepCurve = V_MAX * 0.4f;    // 1.0 m/s
    const float speed_factor_min = 0.8f;       // Min weight scaling
    const float speed_factor_max = 1.5f;       // Max weight scaling
    const float alpha = 0.15f;
    const float YAW_HYST_LOW = 0.08f;
    const float YAW_HYST_HIGH = 0.15f;
    const float YAW_STEEP = 0.35f;
};

#endif // MPC_HPP