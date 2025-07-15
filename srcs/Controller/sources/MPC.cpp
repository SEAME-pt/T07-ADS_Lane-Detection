#include "MPC.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>

MPCController::MPCController(double dt, double L, int N)
    : dt_(dt), L_(L), N_(N) {
    x_ = Eigen::Vector3d::Zero();
    u_ = Eigen::Vector2d::Zero();
    Q_ = Eigen::Matrix3d::Identity();
    Q_(0, 0) = 100.0;
    Q_(1, 1) = 100.0;
    Q_(2, 2) = 10.0;
    R_ = Eigen::Matrix2d::Identity();
    R_(0, 0) = 50.0;
    R_(1, 1) = 20.0;
    S_ = 100.0;
}

void MPCController::update(double ey, double yaw, double v_current) {
    x_(0) = ey;
    x_(1) = yaw;
    x_(2) = v_current;

    Eigen::Vector2d u_opt = u_;
    double delta_prev = u_(0);

    for (int iter = 0; iter < 10; ++iter) {
        Eigen::Vector3d x_pred = x_;
        double cost = 0.0;
        Eigen::Vector2d grad = Eigen::Vector2d::Zero();

        for (int k = 0; k < N_; ++k) {
            x_pred(0) += dt_ * x_pred(2) * std::sin(x_pred(1));
            x_pred(1) += dt_ * (x_pred(2) / L_) * std::tan(u_opt(0));
            x_pred(2) += dt_ * u_opt(1);
            x_pred(2) = std::max(1.0, std::min(2.5, x_pred(2)));

            cost += x_pred.transpose() * Q_ * x_pred;
            cost += u_opt.transpose() * R_ * u_opt;
            if (k > 0) {
                cost += S_ * std::pow(u_opt(0) - delta_prev, 2);
                delta_prev = u_opt(0);
            }

            Eigen::Vector2d u_temp = u_opt;
            double h = 0.01;
            for (int i = 0; i < 2; ++i) {
                u_temp(i) += h;
                double cost_plus = computeCost(x_, u_temp, delta_prev);
                u_temp(i) -= 2 * h;
                double cost_minus = computeCost(x_, u_temp, delta_prev);
                grad(i) = (cost_plus - cost_minus) / (2 * h);
                u_temp(i) = u_opt(i);
            }
        }

        u_opt -= 0.1 * grad;
        u_opt(0) = std::max(-0.5, std::min(0.5, u_opt(0)));
        u_opt(1) = std::max(-1.0, std::min(1.0, u_opt(1)));
        if (std::abs(u_opt(0) - u_(0)) > 0.1) {
            u_opt(0) = u_(0) + std::copysign(0.1, u_opt(0) - u_(0));
        }
    }

    u_ = u_opt;
}

float MPCController::getSteeringAngle() const {
    return static_cast<float>(u_(0));
}

float MPCController::getAcceleration() const {
    return static_cast<float>(u_(1));
}

double MPCController::computeCost(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double delta_prev) {
    Eigen::Vector3d x_pred = x;
    double cost = 0.0;
    for (int k = 0; k < N_; ++k) {
        x_pred(0) += dt_ * x_pred(2) * std::sin(x_pred(1));
        x_pred(1) += dt_ * (x_pred(2) / L_) * std::tan(u(0));
        x_pred(2) += dt_ * u(1);
        x_pred(2) = std::max(1.0, std::min(2.5, x_pred(2)));
        cost += x_pred.transpose() * Q_ * x_pred;
        cost += u.transpose() * R_ * u;
        if (k > 0) {
            cost += S_ * std::pow(u(0) - delta_prev, 2);
            delta_prev = u(0);
        }
    }
    return cost;
}


// use as an example for testing the MPC Controller
// int MPCtester() {
//     // double DT = 0.1;
//     // double L = 0.3;
//     // int N = 10;
//     MPCController mpc(DT, L, N);

//     while (true) {
//         double ey = 0.1;
//         double yaw = 0.05;
//         double v_current = 1.5;

//         mpc.update(ey, yaw, v_current);
//         float delta = mpc.getSteeringAngle();
//         float a = mpc.getAcceleration();

//         // Placeholder functions assumed to be defined elsewhere
//         applySteering(delta);
//         applyAcceleration(a);

//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }
//     return 0;
// }