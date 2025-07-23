#ifndef MPC_CONTROLLER_HPP
#define MPC_CONTROLLER_HPP

#include <Eigen/Dense>

// JetRacer parameters
const double L = 0.15;          // Wheelbase (m)
const double DT = 0.1;        // Time step (s)
const int N = 10;              // Prediction horizon

// JetRacer parameters
const double V_MAX = 2.5;      // Max velocity (m/s)
const double DELTA_MAX = 0.523; // Max steering angle (rad, 30 deg)
const double DELTA_RATE_MAX = 0.2; // Max steering rate (rad/step)
const double A_MAX = 2.0;      // Max acceleration (m/s^2)
const double V_REF = 0.3;      // Reference velocity (m/s)

// MPC weights
const double Q_EY = 100.0;     // Weight for cross-track error
const double Q_PSI_ERR = 10.0; // Weight for heading error
const double Q_V = 1.0;        // Weight for velocity error
const double R_DELTA = 1.0;    // Weight for steering effort
const double R_A = 1.0;        // Weight for acceleration effort

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
};

#endif // MPC_CONTROLLER_HPP