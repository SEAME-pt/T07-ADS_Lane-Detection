#include "nmpc.hpp"
#include <stdexcept>
using namespace casadi;
NMPCController::NMPCController(double L, double dt, int N, double delta_max, double a_max, double w_x, double w_y, double w_psi, double w_v, double w_delta, double w_a)
	: L_(L), dt_(dt), N_(N), delta_max_(delta_max), a_max_(a_max), w_x_(w_x), w_y_(w_y), w_psi_(w_psi), w_v_(w_v), w_delta_(w_delta), w_a_(w_a) {
	// Symbolic variables
	SX x = SX::sym("x");
	SX y = SX::sym("y");
	SX psi = SX::sym("psi");
	SX v = SX::sym("v");
	SX delta = SX::sym("delta");
	SX a = SX::sym("a");
	SX states = vertcat(x, y, psi, v);
	SX controls = vertcat(delta, a);
	// Discretized dynamic model
	SX x_next = x + v * cos(psi) * dt;
	SX y_next = y + v * sin(psi) * dt;
	SX psi_next = psi + (v / L) * tan(delta) * dt;
	SX v_next = v + a * dt;
	SX states_next = vertcat(x_next, y_next, psi_next, v_next);
	// Model function
	Function f("f", {states, controls}, {states_next});
	// Optimization variables
	std::vector<SX> w; // All variables
	std::vector<SX> g; // Constraints
	SX J = 0;		  // Cost function
	std::vector<std::vector<double>> lbw, ubw; // Variable bounds
	std::vector<std::vector<double>> lbg, ubg; // Constraint bounds
	// States and controls over horizon
	std::vector<SX> X(N + 1), U(N);
	for (int k = 0; k < N + 1; ++k) {
		X[k] = SX::sym("X_" + std::to_string(k), 4); // 4 states: x, y, psi, v
		w.push_back(X[k]);
		lbw.push_back({-inf, -inf, -inf, 0.0}); // v >= 0
		ubw.push_back({inf, inf, inf, inf});	// No upper bound on v
	}
	for (int k = 0; k < N; ++k) {
		U[k] = SX::sym("U_" + std::to_string(k), 2); // 2 controls: delta, a
		w.push_back(U[k]);
		lbw.push_back({-delta_max, -a_max}); // Lower bounds for delta, a
		ubw.push_back({delta_max, a_max});   // Upper bounds for delta, a
	}
	// Initial state constraint
	SX x0 = SX::sym("x0", 4);
	g.push_back(X[0] - x0);
	lbg.push_back({0, 0, 0, 0});
	ubg.push_back({0, 0, 0, 0});
	// Dynamics constraints
	for (int k = 0; k < N; ++k) {
		std::vector<SX> f_in = {X[k], U[k]};
		SX xk_next = f(f_in)[0];
		g.push_back(xk_next - X[k + 1]);
		lbg.push_back({0, 0, 0, 0});
		ubg.push_back({0, 0, 0, 0});
	}
	// Cost function
	SX x_ref = SX::sym("x_ref", 2 * N); // Reference trajectory [x_ref_0, y_ref_0, ...]
	SX v_ref = SX::sym("v_ref", N);	 // Reference velocity
	for (int k = 0; k < N; ++k) {
		J += w_x * pow(X[k + 1](0) - x_ref(2 * k), 2);
		J += w_y * pow(X[k + 1](1) - x_ref(2 * k + 1), 2);
		J += w_psi * pow(X[k + 1](2), 2); // psi_ref = 0
		J += w_v * pow(X[k + 1](3) - v_ref(k), 2); // Track reference velocity
		if (k < N) {
			J += w_delta * pow(U[k](0), 2);
			J += w_a * pow(U[k](1), 2);
		}
	}
	// Solver configuration
	Dict opts;
	opts["ipopt.print_level"] = 0;
	opts["ipopt.sb"] = "yes";
	opts["print_time"] = 0;
	opts["ipopt.max_iter"] = 100;
	opts["ipopt.tol"] = 1e-6;
	SXDict nlp = {{"x", vertcat(w)}, {"f", J}, {"g", vertcat(g)}, {"p", vertcat(x0, x_ref, v_ref)}};
	solver_ = nlpsol("solver", "ipopt", nlp, opts);
}
std::vector<double> NMPCController::compute_control(const std::vector<double>& x0, const LaneData& laneData, double psi) {
	if (x0.size() < 4) {
		throw std::runtime_error("Initial state must have at least 4 elements");
	}
	// Reference trajectory
	std::vector<double> x_ref(N_ * 2, 0.0);
	int points_used = std::min(laneData.num_points, N_);
	for (int i = 0; i < points_used; ++i) {
		x_ref[2 * i] = laneData.points[i].x;
		x_ref[2 * i + 1] = laneData.points[i].y;
	}
	for (int i = points_used; i < N_; ++i) {
		x_ref[2 * i] = laneData.points[points_used - 1].x;
		x_ref[2 * i + 1] = laneData.points[points_used - 1].y;
	}
	// Reference velocity (example: constant 3 m/s, adjust as needed)
	std::vector<double> v_ref(N_, 3.0); // Can be dynamic based on laneData or other logic
	// Initial state
	std::vector<double> state = {x0[0], x0[1], psi, x0[3]}; // x, y, psi, v
	// Optimization variables
	std::vector<double> w0((N_ + 1) * 4 + N_ * 2, 0.0);
	std::vector<double> lbx((N_ + 1) * 4 + N_ * 2, 0.0);
	std::vector<double> ubx((N_ + 1) * 4 + N_ * 2, 0.0);
	std::vector<double> lbg((N_ + 1) * 4, 0.0);
	std::vector<double> ubg((N_ + 1) * 4, 0.0);
	// State bounds
	for (int k = 0; k < N_ + 1; ++k) {
		lbx[4 * k] = -inf;
		lbx[4 * k + 1] = -inf;
		lbx[4 * k + 2] = -inf;
		lbx[4 * k + 3] = 0.0; // v >= 0
		ubx[4 * k] = inf;
		ubx[4 * k + 1] = inf;
		ubx[4 * k + 2] = inf;
		ubx[4 * k + 3] = inf;
	}
	// Control bounds
	for (int k = 0; k < N_; ++k) {
		lbx[(N_ + 1) * 4 + 2 * k] = -delta_max_;
		lbx[(N_ + 1) * 4 + 2 * k + 1] = -a_max_;
		ubx[(N_ + 1) * 4 + 2 * k] = delta_max_;
		ubx[(N_ + 1) * 4 + 2 * k + 1] = a_max_;
	}
	// Initial state constraint
	lbg[0] = ubg[0] = state[0];
	lbg[1] = ubg[1] = state[1];
	lbg[2] = ubg[2] = state[2];
	lbg[3] = ubg[3] = state[3];
	// Solver parameters
	std::vector<double> p(4 + 2 * N_ + N_);
	p[0] = state[0];
	p[1] = state[1];
	p[2] = state[2];
	p[3] = state[3];
	for (int i = 0; i < N_; ++i) {
		p[4 + 2 * i] = x_ref[2 * i];
		p[4 + 2 * i + 1] = x_ref[2 * i + 1];
		p[4 + 2 * N_ + i] = v_ref[i];
	}
	DMDict arg = {{"x0", w0}, {"lbx", lbx}, {"ubx", ubx}, {"lbg", lbg}, {"ubg", ubg}, {"p", p}};
	DMDict res = solver_(arg);
	std::vector<double> w_opt(res.at("x").get_elements());
	// Extract first control (delta, a)
	return {w_opt[(N_ + 1) * 4], w_opt[(N_ + 1) * 4 + 1]};
}