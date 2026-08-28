/**
 * @file test_planner_benchmark.cpp
 * @brief Planner 每帧耗时基准（手动运行，不注册默认 CTest）。
 *
 * 分别统计：reference generation + yaw MPC + pitch MPC + total（粗略，
 * 通过多次 plan() 均摊）。allocation 计数留待后续（本轮只测量真实耗时）。
 */

#include "app/auto_aim/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <tuple>
#include <vector>

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

	std::vector<double> aim_us, ref_us, yaw_us, pitch_us, total_us;

	for(int i = 0; i < kIters; ++i)
	{
		auto_aim::PlannerPreviewSeed seed;

		const auto a0 = std::chrono::steady_clock::now();
		aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		const auto a1 = std::chrono::steady_clock::now();

		auto_aim::PlannerDebugData debug;
		const auto p0 = std::chrono::steady_clock::now();
		planner.plan(seed, target, aimer, &debug);
		const auto p1 = std::chrono::steady_clock::now();

		aim_us.push_back(std::chrono::duration<double, std::micro>(a1 - a0).count());
		ref_us.push_back(debug.reference_generation_us);
		yaw_us.push_back(debug.yaw_mpc_us);
		pitch_us.push_back(debug.pitch_mpc_us);
		total_us.push_back(std::chrono::duration<double, std::micro>(p1 - p0).count());
	}

	auto stats = [](std::vector<double>& v) {
		std::sort(v.begin(), v.end());
		const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
		const double p95 = v[static_cast<std::size_t>(v.size() * 0.95)];
		return std::tuple{v.front(), v.back(), mean, p95};
	};

	auto print = [&](const char* name, std::vector<double>& v) {
		auto [mn, mx, mean, p95] = stats(v);
		std::printf("%-22s min=%8.1f  max=%8.1f  mean=%8.1f  p95=%8.1f us\n", name, mn, mx, mean, p95);
	};

	std::printf("Planner Release benchmark (%d iters)\n\n", kIters);
	print("Aimer::aim", aim_us);
	print("reference generation", ref_us);
	print("yaw TinyMPC", yaw_us);
	print("pitch TinyMPC", pitch_us);
	print("Planner total (plan)", total_us);

	return 0;
}
