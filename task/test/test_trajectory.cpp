/**
 * @file test_trajectory.cpp
 * @brief 无空气阻力抛物线弹道求解单元测试。
 *
 * 覆盖：
 * - 正常可解；
 * - h == 0 时低弹道 pitch 应 > 0（受重力影响），fly_time > d/v0；
 * - 高目标 pitch > 0；
 * - 过远 / 无法命中 -> unsolvable，且 fly_time/pitch 默认 0；
 * - 非法速度 v0 <= 0 / NaN / Inf -> unsolvable（safety hardening）；
 * - 取飞行时间较小的低弹道解。
 *
 * expected 值使用 SP25 解析公式独立计算，不直接调用 Trajectory。
 */

#include "tools/trajectory.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

namespace
{

	constexpr double kGravity = 9.7833;

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

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	// ============================================================
	// SP25 解析公式 reference（与 Trajectory 算法一致，独立实现用于对照）
	// ============================================================

	struct AnalyticSolution
	{
		bool solvable = false;
		double pitch = 0.0;
		double fly_time = 0.0;
	};

	AnalyticSolution analytic_solution(double v0, double d, double h)
	{
		AnalyticSolution result;

		if(!std::isfinite(v0) || v0 <= 0.0 || !std::isfinite(d) || d < 0.0 || !std::isfinite(h))
		{
			return result;
		}

		const double a = kGravity * d * d / (2.0 * v0 * v0);
		const double b = -d;
		const double c = a + h;
		const double delta = b * b - 4.0 * a * c;

		if(delta < 0.0)
		{
			return result;
		}

		const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2.0 * a);
		const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2.0 * a);

		const double pitch_1 = std::atan(tan_pitch_1);
		const double pitch_2 = std::atan(tan_pitch_2);

		const double fly_time_1 = d / (v0 * std::cos(pitch_1));
		const double fly_time_2 = d / (v0 * std::cos(pitch_2));

		result.solvable = true;
		result.pitch = (fly_time_1 < fly_time_2) ? pitch_1 : pitch_2;
		result.fly_time = (fly_time_1 < fly_time_2) ? fly_time_1 : fly_time_2;
		return result;
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_normal_solvable(TestRunner& runner)
	{
		runner.begin("Normal solvable");

		const double v0 = 23.0;
		const double d = 3.0;
		const double h = 0.2;

		const tools::Trajectory traj(v0, d, h);
		const AnalyticSolution expected = analytic_solution(v0, d, h);

		runner.expect(!traj.unsolvable, "normal input should be solvable");
		runner.expect(expected.solvable, "analytic reference should be solvable");
		runner.expect(near(traj.pitch, expected.pitch, 1e-12), "pitch should match analytic");
		runner.expect(near(traj.fly_time, expected.fly_time, 1e-12), "fly_time should match analytic");
		runner.expect(traj.fly_time > 0.0, "fly_time should be positive");
		runner.expect(std::isfinite(traj.pitch), "pitch should be finite");

		runner.end();
	}

	void test_flat_level(TestRunner& runner)
	{
		runner.begin("Flat level (h == 0)");

		const double v0 = 23.0;
		const double d = 3.0;
		const double h = 0.0;

		const tools::Trajectory traj(v0, d, h);
		const AnalyticSolution expected = analytic_solution(v0, d, h);

		runner.expect(!traj.unsolvable, "flat level should be solvable");
		// 重力作用下，水平距离为 d 的目标需要轻微抬头，pitch 严格大于 0。
		runner.expect(traj.pitch > 0.0, "flat-level pitch should be > 0 due to gravity");
		// 因为有抬头角，飞行时间略大于无重力直线距离 d/v0。
		runner.expect(traj.fly_time > d / v0, "flat-level fly_time should be > d/v0");
		runner.expect(near(traj.pitch, expected.pitch, 1e-12), "pitch should match analytic");
		runner.expect(near(traj.fly_time, expected.fly_time, 1e-12), "fly_time should match analytic");

		runner.end();
	}

	void test_high_target(TestRunner& runner)
	{
		runner.begin("High target");

		const double v0 = 23.0;
		const double d = 3.0;
		const double h = 1.0;

		const tools::Trajectory traj(v0, d, h);
		const AnalyticSolution expected = analytic_solution(v0, d, h);

		runner.expect(!traj.unsolvable, "high target should be solvable");
		runner.expect(traj.pitch > 0.0, "high target pitch should be > 0");
		runner.expect(near(traj.pitch, expected.pitch, 1e-12), "pitch should match analytic");
		runner.expect(near(traj.fly_time, expected.fly_time, 1e-12), "fly_time should match analytic");

		runner.end();
	}

	void test_impossible_too_far(TestRunner& runner)
	{
		runner.begin("Impossible (too far)");

		const tools::Trajectory traj(15.0, 100.0, 0.0);

		runner.expect(traj.unsolvable, "too-far target should be unsolvable");
		runner.expect(near(traj.fly_time, 0.0), "unsolvable fly_time should default to 0");
		runner.expect(near(traj.pitch, 0.0), "unsolvable pitch should default to 0");

		runner.end();
	}

	void test_invalid_speed(TestRunner& runner)
	{
		runner.begin("Invalid speed (safety hardening)");

		const double nan = std::numeric_limits<double>::quiet_NaN();
		const double inf = std::numeric_limits<double>::infinity();

		const tools::Trajectory zero(0.0, 3.0, 0.0);
		const tools::Trajectory negative(-5.0, 3.0, 0.0);
		const tools::Trajectory nan_traj(nan, 3.0, 0.0);
		const tools::Trajectory inf_traj(inf, 3.0, 0.0);

		runner.expect(zero.unsolvable, "v0 == 0 should be unsolvable");
		runner.expect(negative.unsolvable, "v0 < 0 should be unsolvable");
		runner.expect(nan_traj.unsolvable, "v0 == NaN should be unsolvable");
		runner.expect(inf_traj.unsolvable, "v0 == Inf should be unsolvable");

		runner.end();
	}

	void test_min_fly_time_chosen(TestRunner& runner)
	{
		runner.begin("Minimum fly_time solution chosen");

		const double v0 = 23.0;
		const double d = 5.0;
		const double h = 0.3;

		// 独立计算两解。
		const double a = kGravity * d * d / (2.0 * v0 * v0);
		const double b = -d;
		const double c = a + h;
		const double delta = b * b - 4.0 * a * c;

		runner.expect(delta > 0.0, "test case should have two distinct solutions");

		const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2.0 * a);
		const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2.0 * a);
		const double pitch_1 = std::atan(tan_pitch_1);
		const double pitch_2 = std::atan(tan_pitch_2);
		const double fly_time_1 = d / (v0 * std::cos(pitch_1));
		const double fly_time_2 = d / (v0 * std::cos(pitch_2));
		const double min_fly_time = std::min(fly_time_1, fly_time_2);

		const tools::Trajectory traj(v0, d, h);

		runner.expect(!traj.unsolvable, "should be solvable");
		runner.expect(near(traj.fly_time, min_fly_time, 1e-12), "fly_time should equal min of two solutions");

		runner.end();
	}

	void test_monotonic_distance(TestRunner& runner)
	{
		runner.begin("Monotonic distance");

		// 相同初速/高度，距离越远飞行时间越长（物理 sanity，非重复公式）。
		const tools::Trajectory near_traj(23.0, 2.0, 0.0);
		const tools::Trajectory far_traj(23.0, 5.0, 0.0);

		runner.expect(!near_traj.unsolvable && !far_traj.unsolvable, "both should be solvable");
		runner.expect(far_traj.fly_time > near_traj.fly_time, "farther target should have longer fly_time");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Trajectory Test Suite ===\n\n");

	TestRunner runner;

	test_normal_solvable(runner);
	test_flat_level(runner);
	test_high_target(runner);
	test_impossible_too_far(runner);
	test_invalid_speed(runner);
	test_min_fly_time_chosen(runner);
	test_monotonic_distance(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Trajectory tests failed ===\n");
		return 1;
	}

	std::printf("=== All trajectory tests passed ===\n");
	return 0;
}
