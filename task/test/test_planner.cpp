/**
 * @file test_planner.cpp
 * @brief Planner（TinyMPC 轨迹规划）单元测试。
 *
 * 覆盖：stationary / linear / rotating / token mismatch / deterministic /
 * solver 收敛政策（return code 不作为 valid 判定）/ 加速度有界 / fallback 弹速。
 */

#include "app/auto_aim/planner.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

#include "test_logging.hpp"

namespace
{

	class TestRunner
	{
	public:
		void begin(std::string_view name)
		{
			current_test_ = name;
			current_test_failed_ = false;
			std::printf("===== %.*s =====\n", static_cast<int>(name.size()), name.data());
		}

		void expect(bool condition, std::string_view message)
		{
			++check_count_;

			if(condition)
			{
				std::printf("[PASS] %.*s\n", static_cast<int>(message.size()), message.data());
				return;
			}

			++failure_count_;
			current_test_failed_ = true;
			std::printf("[FAIL] %.*s\n", static_cast<int>(message.size()), message.data());
		}

		void end()
		{
			std::printf("[%s] %.*s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
			            static_cast<int>(current_test_.size()), current_test_.data());
		}

		[[nodiscard]] int failure_count() const noexcept
		{
			return failure_count_;
		}

		void print_summary() const
		{
			std::printf("========================================\n");
			std::printf("Checks:   %d\n", check_count_);
			std::printf("Failures: %d\n", failure_count_);
			std::printf("========================================\n");
		}

	private:
		std::string_view current_test_;
		int check_count_ = 0;
		int failure_count_ = 0;
		bool current_test_failed_ = false;
	};

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	namespace auto_aim = app::auto_aim;

	auto_aim::TrackedTarget make_target(Eigen::Vector3d center, Eigen::Vector3d velocity,
	                                    double yaw, double yaw_rate, double radius, int armor_count,
	                                    std::uint64_t token = 1, bool has_armor_switch = false)
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
		t.target_token = token;
		t.has_armor_switch = has_armor_switch;

		for(int i = 0; i < armor_count; ++i)
		{
			auto_aim::ArmorHypothesis h;
			h.armor_id = i;
			h.position_in_world = Eigen::Vector3d(1.0, 0.0, 0.0);
			h.yaw_in_world = 0.0;
			t.predicted_armors.push_back(h);
		}

		return t;
	}

	auto_aim::PlanningSolution plan_target(auto_aim::Aimer& aimer, auto_aim::Planner& planner,
	                                       const auto_aim::TrackedTarget& target, double bullet_speed)
	{
		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target, 1.0, bullet_speed, nullptr, &seed);
		return planner.plan(seed, target, aimer);
	}

	constexpr double kPi = 3.14159265358979323846;

	void test_stationary(TestRunner& runner)
	{
		runner.begin("Stationary target");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		const auto plan = plan_target(aimer, planner, target, 23.0);

		runner.expect(plan.valid, "plan valid");
		runner.expect(plan.status == auto_aim::PlanningStatus::Success, "status Success");
		runner.expect(std::isfinite(plan.yaw_rad) && std::isfinite(plan.pitch_rad), "finite angles");
		runner.expect(std::abs(plan.yaw_velocity_rad_s) < 0.05, "stationary yaw velocity ~ 0");
		runner.expect(std::abs(plan.pitch_velocity_rad_s) < 0.05, "stationary pitch velocity ~ 0");

		runner.end();
	}

	void test_linear_velocity(TestRunner& runner)
	{
		runner.begin("Constant linear velocity (future trajectory)");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(0.0, 2.0, 0.0),
		                                0.0, 0.0, 0.2, 4);

		const auto plan = plan_target(aimer, planner, target, 23.0);

		runner.expect(plan.valid, "plan valid");
		runner.expect(std::abs(plan.yaw_velocity_rad_s) > 0.1,
		              "moving target -> non-zero yaw velocity (future trajectory used)");

		runner.end();
	}

	void test_rotating(TestRunner& runner)
	{
		runner.begin("Constant yaw-rate rotating");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 1.0, 0.2, 4);

		const auto plan = plan_target(aimer, planner, target, 23.0);

		runner.expect(plan.valid, "plan valid");
		runner.expect(std::isfinite(plan.yaw_rad) && std::isfinite(plan.pitch_rad), "finite angles");
		runner.expect(std::abs(plan.yaw_rad) <= kPi + 1e-6, "yaw within [-pi, pi]");

		runner.end();
	}

	void test_token_mismatch(TestRunner& runner)
	{
		runner.begin("seed/target token mismatch");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target1 = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                 0.0, 0.0, 0.2, 4, 1);
		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target1, 1.0, 23.0, nullptr, &seed);

		const auto target2 = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                 0.0, 0.0, 0.2, 4, 2);
		const auto plan = planner.plan(seed, target2, aimer);

		runner.expect(!plan.valid, "token mismatch -> invalid");
		runner.expect(plan.status == auto_aim::PlanningStatus::InvalidTarget, "status InvalidTarget");

		runner.end();
	}

	void test_deterministic(TestRunner& runner)
	{
		runner.begin("Deterministic cold start");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		const auto p1 = plan_target(aimer, planner, target, 23.0);
		const auto p2 = plan_target(aimer, planner, target, 23.0);

		runner.expect(near(p1.yaw_rad, p2.yaw_rad, 1e-9), "yaw deterministic");
		runner.expect(near(p1.pitch_rad, p2.pitch_rad, 1e-9), "pitch deterministic");
		runner.expect(near(p1.yaw_velocity_rad_s, p2.yaw_velocity_rad_s, 1e-9),
		              "yaw velocity deterministic");

		runner.end();
	}

	void test_solver_policy(TestRunner& runner)
	{
		runner.begin("Solver acceptance policy (return code diagnostic only)");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		const auto plan = plan_target(aimer, planner, target, 23.0);

		runner.expect(plan.valid, "plan valid even if tiny_solve returns non-zero");
		runner.expect(plan.diagnostics.yaw_solve_code >= 0, "yaw solve code recorded");
		runner.expect(plan.diagnostics.pitch_solve_code >= 0, "pitch solve code recorded");

		runner.end();
	}

	void test_acceleration_bound(TestRunner& runner)
	{
		runner.begin("Projected acceleration bound");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(0.0, 2.0, 0.0),
		                                0.0, 0.0, 0.2, 4);

		const auto plan = plan_target(aimer, planner, target, 23.0);

		runner.expect(plan.valid, "plan valid");
		runner.expect(std::abs(plan.yaw_acceleration_rad_s2) <= 50.0 + 1e-9,
		              "projected yaw accel <= max_acc");
		runner.expect(std::abs(plan.pitch_acceleration_rad_s2) <= 100.0 + 1e-9,
		              "projected pitch accel <= max_acc");

		runner.end();
	}

	void test_fallback_bullet(TestRunner& runner)
	{
		runner.begin("Fallback bullet speed propagates to Planner");

		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config()); // Fallback policy
		auto_aim::Planner planner(auto_aim::make_default_planner_config());

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		const double nan = std::numeric_limits<double>::quiet_NaN();
		auto_aim::PlannerPreviewSeed seed;
		const auto aiming = aimer.aim(target, 1.0, nan, nullptr, &seed);
		runner.expect(aiming.valid, "aim valid (fallback)");

		const auto plan = planner.plan(seed, target, aimer);
		runner.expect(plan.valid, "plan valid using effective fallback speed");

		runner.end();
	}
} // namespace

int main()
{
	test_logging::init("test_planner");
	std::printf("=== Planner Test Suite ===\n\n");

	TestRunner runner;
	test_stationary(runner);
	test_linear_velocity(runner);
	test_rotating(runner);
	test_token_mismatch(runner);
	test_deterministic(runner);
	test_solver_policy(runner);
	test_acceleration_bound(runner);
	test_fallback_bullet(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Planner tests failed ===\n");
		return 1;
	}

	std::printf("=== All Planner tests passed ===\n");
	return 0;
}
