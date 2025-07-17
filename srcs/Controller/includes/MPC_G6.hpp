#ifndef NMPC_HPP
#define NMPC_HPP
#include <casadi/casadi.hpp>
#include <vector>
#include "lane.hpp"
class MPCController {
public:
	MPCController(double L, double dt, int N, double delta_max, double a_max, double w_x, double w_y, double w_psi, double w_v, double w_delta, double w_a);
	std::vector<double> compute_control(const std::vector<double>& x0, const LaneData& laneData, double psi);
private:
	double L_;		  // Wheelbase (m)
	double dt_;		 // Time step (s)
	int N_;			 // Prediction horizon
	double delta_max_;  // Max steering angle (rad)
	double a_max_;	  // Max acceleration (m/s^2)
	double w_x_, w_y_, w_psi_, w_v_, w_delta_, w_a_; // Cost function weights
	casadi::Function solver_; // CasADi solver
};
#endif