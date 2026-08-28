/**
 * @file test_tiny_mpc.cpp
 * @brief TinyMpc2d（TinyMPC Robocore 封装）单元测试。
 *
 * 覆盖：
 * - config 校验（非法 dt/rho/r/max_acc/horizon/max_iter 抛异常；Q 允许半正定）；
 * - setup / tracking / 输入约束；
 * - 不可拷贝（compile-time static_assert）；
 * - move stress（构造/析构/move，无 crash；泄漏由 ASan/LSan 单独验证）；
 * - deterministic cold start（不同历史调用后同一输入输出一致）；
 * - 收敛表征（report-only，作为后续 solver policy 冻结的输入）。
 */

#include "tools/tiny_mpc_2d.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "test_logging.hpp"

namespace
{

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

	tools::TinyMpc2d::Config make_config()
	{
		// 默认即 legacy baseline：dt=0.01 rho=1.0 q=[9e6,0] r=1 max_acc=50 horizon=100 max_iter=10
		return tools::TinyMpc2d::Config{};
	}

	Eigen::MatrixXd constant_ref(int horizon, double position, double velocity)
	{
		Eigen::MatrixXd ref(2, horizon);
		ref.row(0).setConstant(position);
		ref.row(1).setConstant(velocity);
		return ref;
	}

	void test_config_validation(TestRunner& runner)
	{
		runner.begin("Config validation");

		const auto throws = [](const tools::TinyMpc2d::Config& c) {
			try
			{
				tools::TinyMpc2d s(c);
				return false;
			}
			catch(const std::invalid_argument&)
			{
				return true;
			}
		};

		{
			auto c = make_config();
			c.dt = 0.0;
			runner.expect(throws(c), "dt == 0 throws");
		}
		{
			auto c = make_config();
			c.rho = 0.0;
			runner.expect(throws(c), "rho == 0 throws");
		}
		{
			auto c = make_config();
			c.r = 0.0;
			runner.expect(throws(c), "r == 0 throws");
		}
		{
			auto c = make_config();
			c.max_acceleration = 0.0;
			runner.expect(throws(c), "max_acceleration == 0 throws");
		}
		{
			auto c = make_config();
			c.horizon = 1;
			runner.expect(throws(c), "horizon < 2 throws");
		}
		{
			auto c = make_config();
			c.max_iter = 0;
			runner.expect(throws(c), "max_iter < 1 throws");
		}

		// Q 允许半正定（legacy velocity weight 本来就是 0）。
		{
			auto c = make_config();
			c.q = Eigen::Vector2d(9e6, 0.0);
			runner.expect(!throws(c), "Q = [9e6, 0] is valid (positive semidefinite)");
		}
		{
			auto c = make_config();
			c.q = Eigen::Vector2d(9e6, -1.0);
			runner.expect(throws(c), "negative Q element throws");
		}

		runner.end();
	}

	void test_setup_tracking_constraint(TestRunner& runner)
	{
		runner.begin("Setup / tracking / constraint");

		tools::TinyMpc2d solver(make_config());
		const int h = solver.horizon();
		runner.expect(h == 100, "horizon == 100");

		const Eigen::MatrixXd ref = constant_ref(h, 0.0, 0.0);
		const Eigen::Vector2d x0(1.0, 0.0);
		solver.solve(x0, ref);

		bool finite_all = true;
		bool acc_bounded = true;

		for(int k = 0; k < h - 1; ++k)
		{
			if(!std::isfinite(solver.position(k)) || !std::isfinite(solver.velocity(k))
			   || !std::isfinite(solver.acceleration(k)))
			{
				finite_all = false;
			}

			if(std::abs(solver.acceleration(k)) > 50.0 + 1e-9)
			{
				acc_bounded = false;
			}
		}

		runner.expect(std::isfinite(solver.position(h - 1)) && std::isfinite(solver.velocity(h - 1)),
		              "tail state finite");
		runner.expect(finite_all, "all outputs finite");
		runner.expect(acc_bounded, "|acceleration| <= max_acc");
		runner.expect(std::abs(solver.position(h - 1)) < std::abs(x0(0)),
		              "position moved toward setpoint");

		runner.end();
	}

	void test_non_copyable(TestRunner& runner)
	{
		runner.begin("Non-copyable (compile-time)");

		static_assert(!std::is_copy_constructible_v<tools::TinyMpc2d>,
		              "TinyMpc2d must not be copy-constructible");
		static_assert(!std::is_copy_assignable_v<tools::TinyMpc2d>,
		              "TinyMpc2d must not be copy-assignable");
		static_assert(std::is_nothrow_move_constructible_v<tools::TinyMpc2d>,
		              "TinyMpc2d should be nothrow move-constructible");
		static_assert(std::is_nothrow_move_assignable_v<tools::TinyMpc2d>,
		              "TinyMpc2d should be nothrow move-assignable");

		runner.expect(true, "static_asserts compiled");
		runner.end();
	}

	void test_move_stress(TestRunner& runner)
	{
		runner.begin("Move stress");

		{
			tools::TinyMpc2d a(make_config());
			tools::TinyMpc2d b(std::move(a));

			Eigen::MatrixXd ref_b = constant_ref(b.horizon(), 0.0, 0.0);
			b.solve(Eigen::Vector2d(1.0, 0.0), ref_b);

			tools::TinyMpc2d c(make_config());
			c = std::move(b);

			Eigen::MatrixXd ref_c = constant_ref(c.horizon(), 0.0, 0.0);
			c.solve(Eigen::Vector2d(0.5, 0.0), ref_c);
			runner.expect(std::isfinite(c.position(c.horizon() - 1)),
			              "moved-to solver produces finite output");
		}

		runner.expect(true, "no crash / double-free in move path (leaks verified under ASan separately)");
		runner.end();
	}

	void test_deterministic_cold_start(TestRunner& runner)
	{
		runner.begin("Deterministic cold start");

		tools::TinyMpc2d solver(make_config());
		const int h = solver.horizon();
		const Eigen::Vector2d x0(0.5, 0.0);

		const Eigen::MatrixXd ref_a = constant_ref(h, 0.0, 0.0);
		solver.solve(x0, ref_a);

		std::vector<double> pos_a(static_cast<std::size_t>(h));
		std::vector<double> vel_a(static_cast<std::size_t>(h));
		std::vector<double> acc_a(static_cast<std::size_t>(h - 1));

		for(int k = 0; k < h; ++k)
		{
			pos_a[static_cast<std::size_t>(k)] = solver.position(k);
			vel_a[static_cast<std::size_t>(k)] = solver.velocity(k);
		}

		for(int k = 0; k < h - 1; ++k)
		{
			acc_a[static_cast<std::size_t>(k)] = solver.acceleration(k);
		}

		// 另一段不同 reference（线性 ramp），干扰 solver 数值状态。
		Eigen::MatrixXd ref_b(2, h);

		for(int k = 0; k < h; ++k)
		{
			ref_b(0, k) = k * 0.01 * 0.5;
			ref_b(1, k) = 0.5;
		}

		solver.solve(x0, ref_b);

		// 再次求解 ref_a，应与第一次结果一致（冷启动）。
		solver.solve(x0, ref_a);

		bool same = true;

		for(int k = 0; k < h; ++k)
		{
			if(!near(solver.position(k), pos_a[static_cast<std::size_t>(k)], 1e-9)
			   || !near(solver.velocity(k), vel_a[static_cast<std::size_t>(k)], 1e-9))
			{
				same = false;
			}
		}

		for(int k = 0; k < h - 1; ++k)
		{
			if(!near(solver.acceleration(k), acc_a[static_cast<std::size_t>(k)], 1e-9))
			{
				same = false;
			}
		}

		runner.expect(same, "same input after different history -> identical output (1e-9)");
		runner.end();
	}

	void test_convergence_characterization(TestRunner& runner)
	{
		runner.begin("Convergence characterization (report-only)");

		const Eigen::Vector2d x0(1.0, 0.0);

		{
			tools::TinyMpc2d solver(make_config()); // max_iter = 10（legacy）
			const int h = solver.horizon();
			const Eigen::MatrixXd ref = constant_ref(h, 0.0, 0.0);
			const int status = solver.solve(x0, ref);
			std::printf("  [info] max_iter=10   -> tiny_solve status=%d, final pos=%.6f\n",
			            status, solver.position(h - 1));
			runner.expect(std::isfinite(solver.position(h - 1)), "max_iter=10 output finite");
		}

		{
			auto c = make_config();
			c.max_iter = 1000;
			tools::TinyMpc2d solver(c);
			const int h = solver.horizon();
			const Eigen::MatrixXd ref = constant_ref(h, 0.0, 0.0);
			const int status = solver.solve(x0, ref);
			std::printf("  [info] max_iter=1000 -> tiny_solve status=%d, final pos=%.6f\n",
			            status, solver.position(h - 1));
			runner.expect(std::isfinite(solver.position(h - 1)), "max_iter=1000 output finite");
		}

		runner.end();
	}

	void test_invalid_input(TestRunner& runner)
	{
		runner.begin("Invalid input fails closed");

		tools::TinyMpc2d solver(make_config());
		const int h = solver.horizon();
		const double nan = std::numeric_limits<double>::quiet_NaN();

		Eigen::Vector2d bad_x0(1.0, 0.0);
		bad_x0(0) = nan;

		const Eigen::MatrixXd ref = constant_ref(h, 0.0, 0.0);
		runner.expect(solver.solve(bad_x0, ref) != 0, "NaN x0 -> rejected (non-zero)");

		Eigen::MatrixXd bad_ref = constant_ref(h, 0.0, 0.0);
		bad_ref(0, 0) = nan;
		runner.expect(solver.solve(Eigen::Vector2d(0.5, 0.0), bad_ref) != 0,
		              "NaN ref -> rejected (non-zero)");

		runner.end();
	}
} // namespace

int main()
{
	test_logging::init("test_tiny_mpc");
	std::printf("=== TinyMpc2d Test Suite ===\n\n");

	TestRunner runner;

	test_config_validation(runner);
	test_setup_tracking_constraint(runner);
	test_non_copyable(runner);
	test_move_stress(runner);
	test_deterministic_cold_start(runner);
	test_convergence_characterization(runner);
	test_invalid_input(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== TinyMpc2d tests failed ===\n");
		return 1;
	}

	std::printf("=== All TinyMpc2d tests passed ===\n");
	return 0;
}
