/**
 * @file test_planner_benchmark.cpp
 * @brief Planner 每帧耗时基准（手动运行，不注册默认 CTest）。
 *
 * 分别统计：reference generation + yaw MPC + pitch MPC + total（粗略，
 * 通过多次 plan() 均摊）。allocation 计数留待后续（本轮只测量真实耗时）。
 */

#include "app/auto_aim/planner.hpp"

#include <chrono>
#include <cstdio>

#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

	auto_aim::TrackedTarget make_target()
	{
		auto_aim::TrackedTarget t;
		t.state = auto_aim::TrackerState::Tracking;
		t.timestamp_s = 0.0;
		t.name = auto_aim::ArmorName::Four;
		t.center_in_world = Eigen::Vector3d(3.0, 0.0, 0.0);
		t.velocity_in_world = Eigen::Vector3d(0.0, 1.0, 0.0);
		t.yaw = 0.0;
		t.yaw_rate = 0.5;
		t.radius = 0.2;
		t.delta_radius = 0.0;
		t.delta_z = 0.0;
		t.target_token = 1;
		t.has_armor_switch = true;

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

} // namespace

int main()
{
	test_logging::init("test_planner_benchmark");

	auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
	auto_aim::Planner planner(auto_aim::make_default_planner_config());
	const auto target = make_target();

	// 预热。
	for(int i = 0; i < 20; ++i)
	{
		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		planner.plan(seed, target, aimer);
	}

	constexpr int kIters = 500;
	const auto t0 = std::chrono::steady_clock::now();

	for(int i = 0; i < kIters; ++i)
	{
		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		planner.plan(seed, target, aimer);
	}

	const auto t1 = std::chrono::steady_clock::now();
	const double total_ms =
	    std::chrono::duration<double, std::milli>(t1 - t0).count();
	const double per_plan_ms = total_ms / kIters;

	std::printf("Planner benchmark: %d iters, total %.2f ms, avg %.3f ms/plan (aim + reference + yaw/pitch MPC)\n",
	            kIters, total_ms, per_plan_ms);

	return 0;
}
