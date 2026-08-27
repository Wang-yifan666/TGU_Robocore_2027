/**
 * @file test_shooter.cpp
 * @brief Shooter 单元测试（纯 synthetic AimingSolution）。
 *
 * 覆盖：auto_fire、invalid、首帧建立历史、两档 tolerance、突变保护、
 * gimbal 未跟上、边界、wrap、NaN/Inf、missing gimbal、负距离、reset、
 * 历史更新时机。
 */

#include "app/auto_aim/shooter.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

	constexpr double kPi = 3.14159265358979323846;

	// ============================================================
	// 简单测试运行器
	// ============================================================

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

	// ============================================================
	// helper
	// ============================================================

	auto_aim::ShooterConfig make_config(bool auto_fire = true, double near_tol = 0.5,
	                                    double far_tol = 0.2, double dist = 3.0)
	{
		auto_aim::ShooterConfig c;
		c.auto_fire = auto_fire;
		c.near_tolerance_rad = near_tol;
		c.far_tolerance_rad = far_tol;
		c.distance_threshold_m = dist;
		return c;
	}

	auto_aim::AimingSolution make_aiming(double yaw, double pitch = 0.0, bool valid = true)
	{
		auto_aim::AimingSolution a;
		a.valid = valid;
		a.yaw_rad = yaw;
		a.pitch_rad = pitch;
		return a;
	}

	double deg(double d)
	{
		return d * kPi / 180.0;
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_auto_fire_disabled(TestRunner& runner)
	{
		runner.begin("auto fire disabled");

		auto_aim::Shooter shooter(make_config(false));
		const auto aiming = make_aiming(0.0);

		runner.expect(!shooter.shoot(aiming, 1.0, 0.0), "disabled -> false");
		runner.expect(!shooter.shoot(aiming, 1.0, 0.0), "disabled stays false");

		runner.end();
	}

	void test_invalid_aiming(TestRunner& runner)
	{
		runner.begin("invalid aiming solution");

		auto_aim::Shooter shooter(make_config(true));
		const auto aiming = make_aiming(0.0, 0.0, false);

		runner.expect(!shooter.shoot(aiming, 1.0, 0.0), "invalid -> false");

		runner.end();
	}

	void test_first_frame_then_fire(TestRunner& runner)
	{
		runner.begin("first frame establishes history");

		auto_aim::Shooter shooter(make_config(true));
		const auto aiming = make_aiming(0.5);

		runner.expect(!shooter.shoot(aiming, 1.0, 0.5), "first frame -> false");
		runner.expect(shooter.shoot(aiming, 1.0, 0.5), "second stable+s settled -> true");

		runner.end();
	}

	void test_far_target_stricter_tolerance(TestRunner& runner)
	{
		runner.begin("far target uses stricter tolerance");

		{
			auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
			const auto aiming = make_aiming(0.0);
			shooter.shoot(aiming, 1.0, 0.0); // establish
			runner.expect(shooter.shoot(aiming, 1.0, 0.3), "near lag 0.3 allowed");
		}

		{
			auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
			const auto aiming = make_aiming(0.0);
			shooter.shoot(aiming, 4.0, 0.0); // establish
			runner.expect(!shooter.shoot(aiming, 4.0, 0.3), "far lag 0.3 denied");
		}

		runner.end();
	}

	void test_yaw_jump_too_large(TestRunner& runner)
	{
		runner.begin("aiming yaw jump too large");

		auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
		shooter.shoot(make_aiming(0.0), 1.0, 0.0); // establish prev = 0

		runner.expect(!shooter.shoot(make_aiming(1.2), 1.0, 0.0), "jump 1.2 > 1.0 denied");

		runner.end();
	}

	void test_gimbal_not_settled(TestRunner& runner)
	{
		runner.begin("gimbal not settled");

		auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
		const auto aiming = make_aiming(0.0);
		shooter.shoot(aiming, 1.0, 0.0); // establish prev = 0

		runner.expect(!shooter.shoot(aiming, 1.0, 0.6), "gimbal lag 0.6 denied");

		runner.end();
	}

	void test_tolerance_boundary(TestRunner& runner)
	{
		runner.begin("tolerance boundary (strict <)");

		{
			auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
			const auto aiming = make_aiming(0.0);
			shooter.shoot(aiming, 1.0, 0.0); // establish
			runner.expect(!shooter.shoot(aiming, 1.0, 0.5), "lag == tol denied");
			runner.expect(shooter.shoot(aiming, 1.0, 0.499999), "lag < tol allowed");
		}

		{
			auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
			shooter.shoot(make_aiming(0.0), 1.0, 0.0); // establish prev = 0
			runner.expect(!shooter.shoot(make_aiming(1.0), 1.0, 0.0), "jump == 2*tol denied");
			runner.expect(shooter.shoot(make_aiming(0.99), 1.0, 1.0), "jump < 2*tol allowed");
		}

		runner.end();
	}

	void test_wrap_around(TestRunner& runner)
	{
		runner.begin("+pi/-pi wrap-around");

		auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
		const double prev = deg(179.0);
		const double curr = deg(-179.0);

		shooter.shoot(make_aiming(prev), 1.0, prev); // establish
		runner.expect(shooter.shoot(make_aiming(curr), 1.0, prev),
		              "+179/-179 (2 deg) should fire, not 358 deg");

		runner.end();
	}

	void test_nan_input(TestRunner& runner)
	{
		runner.begin("NaN input");

		const double nan = std::numeric_limits<double>::quiet_NaN();
		auto_aim::Shooter shooter(make_config(true));

		runner.expect(!shooter.shoot(make_aiming(nan), 1.0, 0.0), "NaN yaw -> false");
		runner.expect(!shooter.shoot(make_aiming(0.0, nan), 1.0, 0.0), "NaN pitch -> false");
		runner.expect(!shooter.shoot(make_aiming(0.0), nan, 0.0), "NaN distance -> false");

		runner.end();
	}

	void test_inf_input(TestRunner& runner)
	{
		runner.begin("Inf input");

		const double inf = std::numeric_limits<double>::infinity();
		auto_aim::Shooter shooter(make_config(true));

		runner.expect(!shooter.shoot(make_aiming(inf), 1.0, 0.0), "Inf yaw -> false");
		runner.expect(!shooter.shoot(make_aiming(0.0), inf, 0.0), "Inf distance -> false");

		runner.end();
	}

	void test_missing_gimbal(TestRunner& runner)
	{
		runner.begin("missing gimbal feedback (NaN sentinel)");

		auto_aim::Shooter shooter(make_config(true));
		const auto aiming = make_aiming(0.5);
		const double nan = std::numeric_limits<double>::quiet_NaN();

		shooter.shoot(aiming, 1.0, 0.5); // establish history
		runner.expect(shooter.shoot(aiming, 1.0, 0.5), "fire true before loss");

		runner.expect(!shooter.shoot(aiming, 1.0, nan), "gimbal NaN -> false");

		// 反馈恢复：第一帧必须重新建立历史，仍禁止开火。
		runner.expect(!shooter.shoot(aiming, 1.0, 0.5), "first frame after recovery no fire");
		runner.expect(shooter.shoot(aiming, 1.0, 0.5), "second frame after recovery fire true");

		runner.end();
	}

	void test_negative_distance(TestRunner& runner)
	{
		runner.begin("negative target distance");

		auto_aim::Shooter shooter(make_config(true));

		runner.expect(!shooter.shoot(make_aiming(0.0), -1.0, 0.0), "negative distance -> false");

		runner.end();
	}

	void test_reset_restores_first_frame(TestRunner& runner)
	{
		runner.begin("reset restores first-frame behavior");

		auto_aim::Shooter shooter(make_config(true));
		const auto aiming = make_aiming(0.5);

		shooter.shoot(aiming, 1.0, 0.5); // establish
		runner.expect(shooter.shoot(aiming, 1.0, 0.5), "fire true before reset");

		shooter.reset();
		runner.expect(!shooter.shoot(aiming, 1.0, 0.5), "first frame after reset -> false");
		runner.expect(shooter.shoot(aiming, 1.0, 0.5), "second frame after reset -> true");

		runner.end();
	}

	void test_invalid_does_not_update_history(TestRunner& runner)
	{
		runner.begin("invalid aim does not update history");

		auto_aim::Shooter shooter(make_config(true));
		const auto aiming = make_aiming(0.0);
		shooter.shoot(aiming, 1.0, 0.0); // establish prev = 0

		runner.expect(!shooter.shoot(make_aiming(0.0, 0.0, false), 1.0, 0.0), "invalid -> false");
		runner.expect(shooter.shoot(aiming, 1.0, 0.0), "history preserved -> true");

		runner.end();
	}

	void test_unstable_updates_history(TestRunner& runner)
	{
		runner.begin("unstable frame still updates history");

		auto_aim::Shooter shooter(make_config(true, 0.5, 0.2, 3.0));
		shooter.shoot(make_aiming(0.0), 1.0, 0.0); // establish prev = 0

		runner.expect(!shooter.shoot(make_aiming(1.2), 1.0, 0.0), "unstable jump denied");
		runner.expect(shooter.shoot(make_aiming(1.2), 1.0, 1.2), "history updated -> true");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_shooter");
	std::printf("=== Shooter Test Suite ===\n\n");

	TestRunner runner;

	test_auto_fire_disabled(runner);
	test_invalid_aiming(runner);
	test_first_frame_then_fire(runner);
	test_far_target_stricter_tolerance(runner);
	test_yaw_jump_too_large(runner);
	test_gimbal_not_settled(runner);
	test_tolerance_boundary(runner);
	test_wrap_around(runner);
	test_nan_input(runner);
	test_inf_input(runner);
	test_missing_gimbal(runner);
	test_negative_distance(runner);
	test_reset_restores_first_frame(runner);
	test_invalid_does_not_update_history(runner);
	test_unstable_updates_history(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Shooter tests failed ===\n");
		return 1;
	}

	std::printf("=== All shooter tests passed ===\n");
	return 0;
}
