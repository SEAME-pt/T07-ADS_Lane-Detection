#ifndef MPC_CONTROLLER_HPP
#define MPC_CONTROLLER_HPP

#include <Eigen/Dense>

// JetRacer parameters
const double L = 0.15;          // Wheelbase (m)
const double DT = 0.5;        // Time step (s)
const int N = 10;              // Prediction horizon

// JetRacer parameters
const double V_MAX = 2.5;      // Max velocity (m/s)
const double DELTA_MAX = 0.523; // Max steering angle (rad, 30 deg)
const double DELTA_RATE_MAX = 0.2; // Max steering rate (rad/step)
const double A_MAX = 2.0;      // Max acceleration (m/s^2)
const double V_REF = 0.7;      // Reference velocity (m/s)

// MPC weights
const double Q_EY = 100.0;     // Weight for cross-track error
const double Q_PSI_ERR = 10.0; // Weight for heading error
const double Q_V = 1.0;        // Weight for velocity error
const double R_DELTA = 1.0;    // Weight for steering effort
const double R_A = 1.0;        // Weight for acceleration effort

class MPCController {
public:
    MPCController(double dt, double L, int N);
    void update(double ey, double yaw, double v_current);
    float getSteeringAngle() const;
    float getAcceleration() const;

private:
    double computeCost(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double delta_prev);
    double dt_;
    double L_;
    int N_;
    Eigen::Vector3d x_;
    Eigen::Vector2d u_;
    Eigen::Matrix3d Q_;
    Eigen::Matrix2d R_;
    double S_;
};

#endif // MPC_CONTROLLER_HPP