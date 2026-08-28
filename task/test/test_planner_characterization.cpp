/**
 * @file test_planner_characterization.cpp
 * @brief TinyMPC primal vs projected constraint fidelity 表征（打印完整数据表）。
 */

#include "app/auto_aim/planner.hpp"

#include <cmath>
#include <cstdio>
#include <string_view>

#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

	auto_aim::TrackedTarget make_target(Eigen::Vector3d center, Eigen::Vector3d velocity,
	                                    double yaw, double yaw_rate, double radius,
	                                    bool has_armor_switch = false)
	{
		auto_aim::TrackedTarget t;
		t.state = auto_aim::TrackerState::Tracking;
		t.timestamp_s = 0.0;
		t.name = auto_aim::ArmorName::Four;
		t.center_in_world = center;
		t.velocity_in_world = velocity;
		t.yaw = yaw;
		t.yaw_rate = yaw_rate;
		t.radius = radius;
		t.delta_radius = 0.0;
		t.delta_z = 0.0;
		t.target_token = 1;
		t.has_armor_switch = has_armor_switch;

		for(int i = 0; i < 4; ++i)
		{
			auto_aim::ArmorHypothesis h;
			h.armor_id = i;
			h.position_in_world = Eigen::Vector3d(1.0, 0.0, 0.0);
			h.yaw_in_world = 0.0;
			t.predicted_armors.push_back(h);
		}

		return t;
	}

	void print_case(const char* name, auto_aim::Aimer& aimer, auto_aim::Planner& planner,
	                const auto_aim::TrackedTarget& target)
	{
		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		const auto p = planner.plan(seed, target, aimer);
		const auto& d = p.diagnostics;

		std::printf("%-14s | yaw: code=%d iter=%d resid=%.3e maxP=%.3f maxPrj=%.3f maxD=%.3f cP=%.3f cPrj=%.3f dC=%.3e\n",
		            name, d.yaw_solve_code, d.yaw_iterations, d.yaw_input_primal_residual,
		            d.yaw_primal_max_abs_acc, d.yaw_projected_max_abs_acc,
		            d.yaw_max_primal_projected_delta, d.yaw_center_primal_u,
		            d.yaw_center_projected_u, d.delta_yaw_center);
		std::printf("%-14s | pit: code=%d iter=%d resid=%.3e maxP=%.3f maxPrj=%.3f maxD=%.3f cP=%.3f cPrj=%.3f dC=%.3e valid=%d\n",
		            "", d.pitch_solve_code, d.pitch_iterations, d.pitch_input_primal_residual,
		            d.pitch_primal_max_abs_acc, d.pitch_projected_max_abs_acc,
		            d.pitch_max_primal_projected_delta, d.pitch_center_primal_u,
		            d.pitch_center_projected_u, d.delta_pitch_center, p.valid ? 1 : 0);
	}

} // namespace

int main()
{
	test_logging::init("test_planner_characterization");

	auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
	auto_aim::Planner planner(auto_aim::make_default_planner_config());

	std::printf("=== Planner constraint fidelity characterization ===\n");
	std::printf("max_yaw_acc=50, max_pitch_acc=100, rho=1, max_iter=10\n\n");

	print_case("stationary", aimer, planner,
	           make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(), 0.0, 0.0, 0.2));
	print_case("linear", aimer, planner,
	           make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(0.0, 2.0, 0.0), 0.0, 0.0, 0.2));
	print_case("rotating", aimer, planner,
	           make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(), 0.0, 1.0, 0.2));
	print_case("armor-switch", aimer, planner,
	           make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(), 0.0, 2.0, 0.2, true));
	print_case("aggressive", aimer, planner,
	           make_target(Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(0.0, 5.0, 0.0), 0.0, 3.0, 0.2, true));

	return 0;
}
