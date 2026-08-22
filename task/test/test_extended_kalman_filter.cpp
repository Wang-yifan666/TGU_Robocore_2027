/**
 * @file test_extended_kalman_filter.cpp
 * @brief 通用 ExtendedKalmanFilter 单元测试。
 *
 * 全部 deterministic：无相机 / 无 OpenVINO / 无视频 / 无随机依赖。
 * 覆盖解析答案、维度/有限性/对称性契约、数值失败、diagnostics 生命周期。
 */

#include "tools/extended_kalman_filter.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace
{

	constexpr double kPi = 3.14159265358979323846;

	class TestRunner
	{
	public:
		void begin(const std::string& name)
		{
			current_test_ = name;
			current_test_failed_ = false;

			std::printf("===== %s =====\n", name.c_str());
		}

		void expect(bool condition, const std::string& message)
		{
			++check_count_;

			if(condition)
			{
				std::printf("[PASS] %s\n", message.c_str());
				return;
			}

			++failure_count_;
			current_test_failed_ = true;

			std::printf("[FAIL] %s\n", message.c_str());
		}

		void end()
		{
			std::printf("[%s] %s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
			            current_test_.c_str());
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
		std::string current_test_;

		int check_count_ = 0;
		int failure_count_ = 0;

		bool current_test_failed_ = false;
	};

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	bool vec_near(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs, double eps = 1e-9)
	{
		return lhs.size() == rhs.size() && (lhs - rhs).norm() <= eps;
	}

	bool mat_near(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs, double eps = 1e-9)
	{
		return lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()
		    && (lhs - rhs).norm() <= eps;
	}

	Eigen::VectorXd vec1(double a)
	{
		Eigen::VectorXd v(1);
		v << a;
		return v;
	}

	Eigen::MatrixXd mat11(double a)
	{
		Eigen::MatrixXd m(1, 1);
		m << a;
		return m;
	}

	// wrap angle 到 [-pi, pi)
	double wrap_angle(double angle)
	{
		constexpr double two_pi = 2.0 * kPi;

		double wrapped = std::fmod(angle + kPi, two_pi);

		if(wrapped < 0.0)
		{
			wrapped += two_pi;
		}

		return wrapped - kPi;
	}

	// ============================================================
	// Test 1：constructor / dimensions / reset
	// ============================================================

	void test_constructor_dimensions(TestRunner& runner)
	{
		runner.begin("Constructor / dimensions");

		Eigen::VectorXd x0(2);
		x0 << 1.0, 2.0;

		Eigen::MatrixXd p0 = 2.0 * Eigen::MatrixXd::Identity(2, 2);

		tools::ExtendedKalmanFilter ekf(x0, p0);

		runner.expect(vec_near(ekf.state(), x0), "state matches x0");
		runner.expect(mat_near(ekf.covariance(), p0), "covariance matches P0");
		runner.expect(ekf.state_dim() == 2, "state_dim == 2");
		runner.expect(ekf.last_innovation().size() == 0,
		              "last_innovation is empty after construction");
		runner.expect(std::isnan(ekf.last_nis()), "last_nis is NaN after construction");

		// 非法 P0 维度（3x3 配 2 维 state）。
		{
			Eigen::MatrixXd bad_p0 = Eigen::MatrixXd::Identity(3, 3);
			bool threw = false;

			try
			{
				tools::ExtendedKalmanFilter bad(x0, bad_p0);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "invalid P0 dimension throws invalid_argument");
		}

		// 非法 P0（含 NaN）。
		{
			Eigen::MatrixXd bad_p0 = Eigen::MatrixXd::Zero(2, 2);
			bad_p0(0, 0) = std::numeric_limits<double>::quiet_NaN();
			bool threw = false;

			try
			{
				tools::ExtendedKalmanFilter bad(x0, bad_p0);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "non-finite P0 throws invalid_argument");
		}

		// reset：state/cov 替换，diagnostics 清空。
		{
			Eigen::VectorXd x1(2);
			x1 << 5.0, 6.0;

			Eigen::MatrixXd p1 = Eigen::MatrixXd::Identity(2, 2);

			ekf.reset(x1, p1);

			runner.expect(vec_near(ekf.state(), x1), "reset replaces state");
			runner.expect(mat_near(ekf.covariance(), p1), "reset replaces covariance");
			runner.expect(ekf.last_innovation().size() == 0,
			              "reset clears last_innovation");
			runner.expect(std::isnan(ekf.last_nis()), "reset resets last_nis to NaN");
		}

		runner.end();
	}

	// ============================================================
	// Test 2：scalar linear update（解析答案）
	// ============================================================

	void test_scalar_linear_update(TestRunner& runner)
	{
		runner.begin("Scalar linear update");

		tools::ExtendedKalmanFilter ekf(vec1(0.0), mat11(1.0));

		const bool ok = ekf.update(vec1(1.0), mat11(1.0), mat11(1.0));

		runner.expect(ok, "update returns true");

		// 理论：S = 2, K = 0.5, x_post = 0.5, P_post = 0.5。
		runner.expect(near(ekf.state()(0), 0.5), "x_post == 0.5");
		runner.expect(near(ekf.covariance()(0, 0), 0.5), "P_post == 0.5");

		// innovation = 1, NIS = 1^2 / 2 = 0.5。
		runner.expect(ekf.last_innovation().size() == 1,
		              "last_innovation has size 1");
		runner.expect(near(ekf.last_innovation()(0), 1.0), "innovation == 1");
		runner.expect(near(ekf.last_nis(), 0.5), "NIS == 0.5");

		runner.end();
	}

	// ============================================================
	// Test 3：linear predict（constant velocity）
	// ============================================================

	void test_linear_predict(TestRunner& runner)
	{
		runner.begin("Linear predict");

		Eigen::VectorXd x0(2);
		x0 << 0.0, 1.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd F(2, 2);
		F << 1.0, 1.0, 0.0, 1.0;

		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
		Q(0, 0) = 0.1;
		Q(1, 1) = 0.1;

		tools::ExtendedKalmanFilter ekf(x0, p0);

		ekf.predict(F, Q);

		Eigen::VectorXd expected_x = F * x0;
		Eigen::MatrixXd expected_p = F * p0 * F.transpose() + Q;

		runner.expect(vec_near(ekf.state(), expected_x), "x_pred == F * x");
		runner.expect(mat_near(ekf.covariance(), expected_p, 1e-12),
		              "P_pred == F P F^T + Q");

		runner.end();
	}

	// ============================================================
	// Test 4：nonlinear predict
	// ============================================================

	void test_nonlinear_predict(TestRunner& runner)
	{
		runner.begin("Nonlinear predict");

		Eigen::VectorXd x0(2);
		x0 << 0.5, 1.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd Q = 0.1 * Eigen::MatrixXd::Identity(2, 2);

		// f(x) = [a + sin(b), b]
		auto f = [](const Eigen::VectorXd& x) {
			Eigen::VectorXd y(2);
			y << x(0) + std::sin(x(1)), x(1);
			return y;
		};

		// Jacobian F = [[1, cos(b)], [0, 1]]，在 x_prior 处求值。
		Eigen::MatrixXd F(2, 2);
		F << 1.0, std::cos(x0(1)), 0.0, 1.0;

		tools::ExtendedKalmanFilter ekf(x0, p0);

		ekf.predict(F, Q, f);

		Eigen::VectorXd expected_x = f(x0);
		Eigen::MatrixXd expected_p = F * p0 * F.transpose() + Q;

		runner.expect(vec_near(ekf.state(), expected_x), "state uses f(x_prior)");
		runner.expect(mat_near(ekf.covariance(), expected_p, 1e-12),
		              "covariance uses F P F^T + Q");

		runner.end();
	}

	// ============================================================
	// Test 5：nonlinear measurement（完整解析 posterior）
	// ============================================================

	void test_nonlinear_measurement(TestRunner& runner)
	{
		runner.begin("Nonlinear measurement (analytic posterior)");

		Eigen::VectorXd x0(2);
		x0 << 3.0, 4.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd H(1, 2);
		H << 0.6, 0.8;

		Eigen::MatrixXd R(1, 1);
		R << 1.0;

		Eigen::VectorXd z(1);
		z << 6.0;

		// h(x) = sqrt(x^2 + y^2)
		auto h = [](const Eigen::VectorXd& x) {
			Eigen::VectorXd y(1);
			y << std::sqrt(x(0) * x(0) + x(1) * x(1));
			return y;
		};

		tools::ExtendedKalmanFilter ekf(x0, p0);

		const bool ok = ekf.update(z, H, R, h);

		runner.expect(ok, "update returns true");

		Eigen::VectorXd expected_x(2);
		expected_x << 3.3, 4.4;

		Eigen::MatrixXd expected_p(2, 2);
		expected_p << 0.82, -0.24, -0.24, 0.68;

		runner.expect(vec_near(ekf.state(), expected_x), "x_post == [3.3, 4.4]");
		runner.expect(mat_near(ekf.covariance(), expected_p),
		              "P_post == [[0.82,-0.24],[-0.24,0.68]]");

		runner.expect(near(ekf.last_innovation()(0), 1.0), "innovation == 1");
		runner.expect(near(ekf.last_nis(), 0.5), "NIS == 0.5");

		runner.end();
	}

	// ============================================================
	// Test 6：angle residual hook
	// ============================================================

	void test_angle_residual(TestRunner& runner)
	{
		runner.begin("Angle residual hook");

		// predicted angle = +pi - 0.01
		const double predicted = kPi - 0.01;
		// measurement = -pi + 0.01
		const double measurement = -kPi + 0.01;

		// 默认 subtraction 会是约 -2pi + 0.02；wrapped residual 应为 +0.02。
		auto residual = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			Eigen::VectorXd r(1);
			r << wrap_angle(a(0) - b(0));
			return r;
		};

		tools::ExtendedKalmanFilter ekf(vec1(predicted), mat11(1.0),
		                                [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			                                return a + b;
		                                },
		                                residual);

		const bool ok = ekf.update(vec1(measurement), mat11(1.0), mat11(1.0));

		runner.expect(ok, "update returns true");
		runner.expect(near(ekf.last_innovation()(0), 0.02, 1e-6),
		              "wrapped residual is ~0.02 rad");

		// 校正实际使用 wrapped +0.02，而不是裸 -2pi + 0.02：
		// K = 0.5, correction = 0.01, x_post = (pi - 0.01) + 0.01 = pi。
		// 若错误使用约 -2pi 的裸残差，posterior 会趋近 0。
		runner.expect(near(ekf.state()(0), kPi, 1e-9),
		              "posterior used wrapped +0.02 residual (state == pi)");

		runner.end();
	}

	// ============================================================
	// Test 7：custom state addition（角度 normalize）
	// ============================================================

	void test_custom_state_addition(TestRunner& runner)
	{
		runner.begin("Custom state addition");

		bool hook_called = false;

		auto state_add = [&hook_called](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			hook_called = true;

			Eigen::VectorXd r(1);
			r << wrap_angle(a(0) + b(0));
			return r;
		};

		// x_prior = 3.0 rad；一次 update 后 correction 会把角度推出 [-pi, pi)。
		tools::ExtendedKalmanFilter ekf(vec1(3.0), mat11(1.0), state_add,
		                                [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			                                return a - b;
		                                });

		// z = 3.5 → innovation = 0.5, S = 2, K = 0.5, correction = 0.25。
		const bool ok = ekf.update(vec1(3.5), mat11(1.0), mat11(1.0));

		runner.expect(ok, "update returns true");
		runner.expect(hook_called, "state_add hook was invoked");

		const double expected = wrap_angle(3.0 + 0.25);

		runner.expect(near(ekf.state()(0), expected, 1e-9),
		              "state angle normalized to [-pi, pi)");
		runner.expect(ekf.state()(0) >= -kPi && ekf.state()(0) < kPi,
		              "state angle within [-pi, pi)");

		runner.end();
	}

	// ============================================================
	// Test 8：covariance health
	// ============================================================

	void test_covariance_health(TestRunner& runner)
	{
		runner.begin("Covariance health");

		Eigen::VectorXd x0(2);
		x0 << 1.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd F(2, 2);
		F << 1.0, 0.1, 0.0, 1.0;

		Eigen::MatrixXd Q = 0.01 * Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd H(1, 2);
		H << 1.0, 0.0;

		Eigen::MatrixXd R(1, 1);
		R << 0.1;

		tools::ExtendedKalmanFilter ekf(x0, p0);

		for(int i = 0; i < 20; ++i)
		{
			ekf.predict(F, Q);
			ekf.update(vec1(1.0 + 0.1 * i), H, R);
		}

		const Eigen::MatrixXd& p = ekf.covariance();

		runner.expect(p.allFinite(), "covariance all finite");
		runner.expect((p - p.transpose()).norm() < 1e-9, "covariance symmetric");
		runner.expect(p.diagonal().minCoeff() >= -1e-9,
		              "covariance diagonal >= small negative tolerance");

		runner.end();
	}

	// ============================================================
	// Test：完整流程（确定性匀速轨迹，多步 predict + update）
	// ============================================================

	void test_full_flow_trajectory(TestRunner& runner)
	{
		runner.begin("Full flow trajectory");

		const double dt = 0.1;
		const double true_velocity = 5.0;

		// 状态 [position, velocity]，业务无关的 constant-velocity 模型。
		Eigen::VectorXd x0(2);
		x0 << 0.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd F(2, 2);
		F << 1.0, dt, 0.0, 1.0;

		Eigen::MatrixXd Q = 0.01 * Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd H(1, 2);
		H << 1.0, 0.0;

		Eigen::MatrixXd R(1, 1);
		R << 0.1;

		tools::ExtendedKalmanFilter ekf(x0, p0);

		const int steps = 50;
		double true_position = 0.0;
		bool all_updates_ok = true;

		for(int k = 0; k < steps; ++k)
		{
			// 真实轨迹确定性推进，无随机噪声。
			true_position = true_velocity * static_cast<double>(k + 1) * dt;

			ekf.predict(F, Q);

			if(!ekf.update(vec1(true_position), H, R))
			{
				all_updates_ok = false;
			}
		}

		runner.expect(all_updates_ok, "all updates succeed in full flow");

		const double position_error = std::abs(ekf.state()(0) - true_position);
		const double velocity_error = std::abs(ekf.state()(1) - true_velocity);

		runner.expect(ekf.state().allFinite(), "final state finite");
		runner.expect(position_error < 1.0, "position converged near true value");
		runner.expect(velocity_error < 1.0, "velocity converged near true value");
		runner.expect(position_error < std::abs(true_position),
		              "position error reduced relative to initial");

		const Eigen::MatrixXd& p = ekf.covariance();
		runner.expect(p.allFinite(), "final covariance finite");
		runner.expect((p - p.transpose()).norm() < 1e-9,
		              "final covariance symmetric");
		runner.expect(p.diagonal().minCoeff() >= -1e-9,
		              "final covariance diagonal >= small negative tolerance");

		runner.end();
	}

	// ============================================================
	// Test 9：dimension mismatch
	// ============================================================

	void test_dimension_mismatch(TestRunner& runner)
	{
		runner.begin("Dimension mismatch");

		Eigen::VectorXd x0(2);
		x0 << 0.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		tools::ExtendedKalmanFilter ekf(x0, p0);

		// H columns != n。
		{
			Eigen::MatrixXd H(1, 3);
			H << 1.0, 0.0, 0.0;

			bool threw = false;

			try
			{
				ekf.update(vec1(1.0), H, mat11(1.0));
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "H with wrong columns throws invalid_argument");
		}

		// Q 维度 mismatch。
		{
			Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(3, 3);
			bool threw = false;

			try
			{
				ekf.predict(Eigen::MatrixXd::Identity(2, 2), Q);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "Q with wrong dimension throws invalid_argument");
		}

		// R 维度 mismatch。
		{
			Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2);
			bool threw = false;

			try
			{
				ekf.update(vec1(1.0), Eigen::MatrixXd::Ones(1, 2), R);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "R with wrong dimension throws invalid_argument");
		}

		// callback h() 返回错误 measurement 维度。
		{
			auto bad_h = [](const Eigen::VectorXd&) {
				Eigen::VectorXd y(2);
				y << 1.0, 2.0;
				return y;
			};

			bool threw = false;

			try
			{
				ekf.update(vec1(1.0), Eigen::MatrixXd::Ones(1, 2), mat11(1.0), bad_h);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "h() returning wrong dimension throws invalid_argument");
		}

		runner.end();
	}

	// ============================================================
	// Test 10：NIS（使用 Test 2 解析答案）
	// ============================================================

	void test_nis(TestRunner& runner)
	{
		runner.begin("NIS");

		tools::ExtendedKalmanFilter ekf(vec1(0.0), mat11(1.0));

		ekf.update(vec1(1.0), mat11(1.0), mat11(1.0));

		// innovation = 1, S = 2 → NIS = 0.5。
		runner.expect(near(ekf.last_nis(), 0.5), "NIS == 0.5 for scalar example");

		runner.end();
	}

	// ============================================================
	// Test：numerical failure（singular S）
	// ============================================================

	void test_numerical_failure(TestRunner& runner)
	{
		runner.begin("Numerical failure (singular S)");

		// x = [0], P = [0], H = [1], R = [0], z = [1] → S = 0 奇异。
		tools::ExtendedKalmanFilter ekf(vec1(0.0), mat11(0.0));

		const Eigen::VectorXd old_state = ekf.state();
		const Eigen::MatrixXd old_covariance = ekf.covariance();
		const Eigen::VectorXd old_innovation = ekf.last_innovation();
		const double old_nis = ekf.last_nis();

		const bool ok = ekf.update(vec1(1.0), mat11(1.0), mat11(0.0));

		runner.expect(!ok, "singular S makes update return false");
		runner.expect(vec_near(ekf.state(), old_state), "state unchanged on failure");
		runner.expect(mat_near(ekf.covariance(), old_covariance),
		              "covariance unchanged on failure");
		runner.expect(ekf.last_innovation().size() == old_innovation.size(),
		              "last_innovation unchanged on failure");
		runner.expect(std::isnan(ekf.last_nis()) == std::isnan(old_nis),
		              "last_nis unchanged on failure (still NaN)");

		runner.end();
	}

	// ============================================================
	// Test：linear predict arithmetic failure（F*x 溢出）
	// ============================================================

	void test_linear_predict_arithmetic_failure(TestRunner& runner)
	{
		runner.begin("Linear predict arithmetic failure");

		// 全部输入合法 finite：x0=[1e308], P0=[1], F=[1e308], Q=[0]。
		// F*x = 1e308*1e308 = +inf，是内部算术溢出（非 callback 错误）。
		Eigen::VectorXd x0(1);
		x0 << 1e308;

		Eigen::MatrixXd p0 = mat11(1.0);
		Eigen::MatrixXd F = mat11(1e308);
		Eigen::MatrixXd Q = mat11(0.0);

		tools::ExtendedKalmanFilter ekf(x0, p0);

		bool threw_runtime = false;
		bool threw_invalid = false;

		try
		{
			ekf.predict(F, Q);
		}
		catch(const std::runtime_error&)
		{
			threw_runtime = true;
		}
		catch(const std::invalid_argument&)
		{
			threw_invalid = true;
		}

		runner.expect(threw_runtime, "linear overflow throws runtime_error");
		runner.expect(!threw_invalid, "linear overflow does not throw invalid_argument");

		// 完整 rollback：state/covariance/diagnostics 全不变。
		runner.expect(vec_near(ekf.state(), x0), "state unchanged after overflow");
		runner.expect(mat_near(ekf.covariance(), p0), "covariance unchanged after overflow");
		runner.expect(ekf.last_innovation().size() == 0,
		              "last_innovation unchanged after overflow");
		runner.expect(std::isnan(ekf.last_nis()), "last_nis unchanged after overflow (NaN)");

		runner.end();
	}

	// ============================================================
	// Test：nonlinear callback bad output 仍为 invalid_argument
	// ============================================================

	void test_nonlinear_predict_callback_bad_output(TestRunner& runner)
	{
		runner.begin("Nonlinear predict callback bad output");

		Eigen::VectorXd x0(2);
		x0 << 0.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);
		Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2, 2);
		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);

		auto bad_f = [](const Eigen::VectorXd&) {
			Eigen::VectorXd y(2);
			y << std::numeric_limits<double>::quiet_NaN(), 0.0;
			return y;
		};

		tools::ExtendedKalmanFilter ekf(x0, p0);

		bool threw_invalid = false;

		try
		{
			ekf.predict(F, Q, bad_f);
		}
		catch(const std::invalid_argument&)
		{
			threw_invalid = true;
		}

		runner.expect(threw_invalid,
		              "nonlinear f() returning NaN throws invalid_argument");

		runner.end();
	}

	// ============================================================
	// Test：diagnostics 在成功 update -> predict 后保留
	// ============================================================

	void test_diagnostics_retained_after_predict(TestRunner& runner)
	{
		runner.begin("Diagnostics retained after predict");

		tools::ExtendedKalmanFilter ekf(vec1(0.0), mat11(1.0));

		// 成功 update：innovation=1, NIS=0.5。
		const bool ok = ekf.update(vec1(1.0), mat11(1.0), mat11(1.0));
		runner.expect(ok, "initial update succeeds");

		// predict 不应刷新 diagnostics。
		ekf.predict(mat11(1.0), mat11(0.1));

		runner.expect(ekf.last_innovation().size() == 1,
		              "last_innovation retained after predict");
		runner.expect(near(ekf.last_innovation()(0), 1.0),
		              "innovation value retained after predict");
		runner.expect(near(ekf.last_nis(), 0.5), "NIS retained after predict");

		runner.end();
	}

	// ============================================================
	// Test：成功 update -> 失败 update 后 diagnostics 完整保留
	// ============================================================

	void test_diagnostics_retained_after_failed_update(TestRunner& runner)
	{
		runner.begin("Diagnostics retained after failed update");

		Eigen::VectorXd x0(2);
		x0 << 0.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		Eigen::MatrixXd H_good(1, 2);
		H_good << 1.0, 0.0;

		tools::ExtendedKalmanFilter ekf(x0, p0);

		// 成功 update（H=[1,0], R=[1], z=[1]）→ P=[[0.5,0],[0,1]], innov=[1], NIS=0.5。
		const bool ok = ekf.update(vec1(1.0), H_good, mat11(1.0));
		runner.expect(ok, "good update succeeds");

		const Eigen::VectorXd good_state = ekf.state();
		const Eigen::MatrixXd good_covariance = ekf.covariance();
		const Eigen::VectorXd good_innovation = ekf.last_innovation();
		const double good_nis = ekf.last_nis();

		// 失败的 update：S = [[1.5,0],[0,0]] 奇异。
		Eigen::VectorXd z_bad(2);
		z_bad << 1.0, 1.0;

		Eigen::MatrixXd H_bad(2, 2);
		H_bad << 1.0, 0.0, 0.0, 0.0;

		Eigen::MatrixXd R_bad = Eigen::MatrixXd::Zero(2, 2);
		R_bad(0, 0) = 1.0;

		const bool bad_ok = ekf.update(z_bad, H_bad, R_bad);

		runner.expect(!bad_ok, "singular-S update returns false");
		runner.expect(vec_near(ekf.state(), good_state),
		              "state retains previous successful update");
		runner.expect(mat_near(ekf.covariance(), good_covariance),
		              "covariance retains previous successful update");
		runner.expect(ekf.last_innovation().size() == good_innovation.size(),
		              "last_innovation size unchanged");
		runner.expect(
		    good_innovation.size() == 0
		        || near(ekf.last_innovation()(0), good_innovation(0)),
		    "last_innovation value unchanged");
		runner.expect(near(ekf.last_nis(), good_nis), "NIS unchanged");

		runner.end();
	}

	// ============================================================
	// Test：空 callback 在构造阶段被拒绝
	// ============================================================

	void test_callback_callability(TestRunner& runner)
	{
		runner.begin("Callback callability");

		Eigen::VectorXd x0(2);
		x0 << 0.0, 0.0;

		Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(2, 2);

		// 空 state_add 立即 invalid_argument。
		{
			bool threw = false;

			try
			{
				tools::ExtendedKalmanFilter ekf(
				    x0, p0, tools::ExtendedKalmanFilter::StateAddFn{},
				    [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
					    return a + b;
				    });
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "empty state_add throws invalid_argument at construction");
		}

		// 空 residual 立即 invalid_argument。
		{
			bool threw = false;

			try
			{
				tools::ExtendedKalmanFilter ekf(
				    x0, p0,
				    [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
					    return a + b;
				    },
				    tools::ExtendedKalmanFilter::ResidualFn{});
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "empty residual throws invalid_argument at construction");
		}

		// 默认 callback 仍正常。
		{
			bool threw = false;

			try
			{
				tools::ExtendedKalmanFilter ekf(x0, p0);
			}
			catch(...)
			{
				threw = true;
			}

			runner.expect(!threw, "default callbacks still construct successfully");
		}

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== ExtendedKalmanFilter Module Test Suite ===\n\n");

	TestRunner runner;

	test_constructor_dimensions(runner);
	test_scalar_linear_update(runner);
	test_linear_predict(runner);
	test_nonlinear_predict(runner);
	test_nonlinear_measurement(runner);
	test_angle_residual(runner);
	test_custom_state_addition(runner);
	test_covariance_health(runner);
	test_full_flow_trajectory(runner);
	test_dimension_mismatch(runner);
	test_nis(runner);
	test_numerical_failure(runner);
	test_linear_predict_arithmetic_failure(runner);
	test_nonlinear_predict_callback_bad_output(runner);
	test_diagnostics_retained_after_predict(runner);
	test_diagnostics_retained_after_failed_update(runner);
	test_callback_callability(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== ExtendedKalmanFilter tests failed ===\n");
		return 1;
	}

	std::printf("=== All extended_kalman_filter tests passed ===\n");
	return 0;
}