/**
 * @file test_target.cpp
 * @brief Target 车辆模型（11D）确定性单元测试。
 *
 * 无相机 / 无 OpenVINO / 无视频 / 无随机依赖。
 * 覆盖状态布局、初始化、armor-count 规则、预测、几何、测量模型与解析 Jacobian。
 */

#include "app/auto_aim/target.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "tools/maths_tools.hpp"

namespace
{

	constexpr double kPi = 3.14159265358979323846;
	constexpr double kTwoPi = 2.0 * kPi;

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

	app::auto_aim::TargetModelConfig deterministic_config()
	{
		app::auto_aim::TargetModelConfig c;
		c.translation_accel_variance = 1.0;
		c.yaw_accel_variance = 1.0;
		c.radius_random_walk_variance = 1.0;
		c.delta_radius_random_walk_variance = 1.0;
		c.delta_z_random_walk_variance = 1.0;
		return c;
	}

	app::auto_aim::ArmorObservation make_observation(double x, double y, double z, double yaw,
	                                                 app::auto_aim::ArmorColor color,
	                                                 app::auto_aim::ArmorName name,
	                                                 app::auto_aim::ArmorType type)
	{
		app::auto_aim::ArmorObservation o;
		o.color = color;
		o.name = name;
		o.type = type;
		o.position_in_world = Eigen::Vector3d(x, y, z);
		o.armor_yaw_in_world = yaw;
		return o;
	}

	app::auto_aim::Target make_target(double center_x, double center_y, double center_z,
	                                  double yaw, double radius)
	{
		// 从观测反向构造：观测位置 = center - radius*(cos(yaw),sin(yaw))。
		const double obs_x = center_x - radius * std::cos(yaw);
		const double obs_y = center_y - radius * std::sin(yaw);

		auto o = make_observation(obs_x, obs_y, center_z, yaw, app::auto_aim::ArmorColor::Red,
		                          app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
		                                              app::auto_aim::kTargetStateDim);

		return app::auto_aim::Target(o, radius, P0, deterministic_config());
	}

	// ============================================================
	// Test：state dimension
	// ============================================================

	void test_state_dimension(TestRunner& runner)
	{
		runner.begin("State dimension");

		runner.expect(app::auto_aim::kTargetStateDim == 11, "state dimension == 11");

		const app::auto_aim::Target target = make_target(0.0, 0.0, 0.0, 0.0, 0.2);

		runner.expect(target.state().size() == 11, "target state size == 11");
		runner.expect(target.covariance().rows() == 11, "covariance rows == 11");
		runner.expect(target.covariance().cols() == 11, "covariance cols == 11");

		runner.end();
	}

	// ============================================================
	// Test：initialization center geometry
	// ============================================================

	void test_initialization_geometry(TestRunner& runner)
	{
		runner.begin("Initialization geometry");

		const double radius = 0.25;
		const double observed_yaw = 0.7;

		auto o = make_observation(1.0, 2.0, 3.0, observed_yaw, app::auto_aim::ArmorColor::Blue,
		                          app::auto_aim::ArmorName::Outpost,
		                          app::auto_aim::ArmorType::Big);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
		                                              app::auto_aim::kTargetStateDim);

		app::auto_aim::Target target(o, radius, P0, deterministic_config());

		const Eigen::VectorXd& x = target.state();

		const double expected_cx = 1.0 + radius * std::cos(observed_yaw);
		const double expected_cy = 2.0 + radius * std::sin(observed_yaw);

		runner.expect(near(x(app::auto_aim::kStateX), expected_cx), "center_x computed");
		runner.expect(near(x(app::auto_aim::kStateY), expected_cy), "center_y computed");
		runner.expect(near(x(app::auto_aim::kStateZ), 3.0), "center_z == observed_z");
		runner.expect(near(x(app::auto_aim::kStateYaw), observed_yaw), "yaw == observed_yaw");
		runner.expect(near(x(app::auto_aim::kStateRadius), radius), "radius == initial radius");

		// 初始速度/yaw_rate/delta 均为 0。
		runner.expect(near(x(app::auto_aim::kStateVx), 0.0), "vx == 0");
		runner.expect(near(x(app::auto_aim::kStateVy), 0.0), "vy == 0");
		runner.expect(near(x(app::auto_aim::kStateVz), 0.0), "vz == 0");
		runner.expect(near(x(app::auto_aim::kStateYawRate), 0.0), "yaw_rate == 0");
		runner.expect(near(x(app::auto_aim::kStateDeltaRadius), 0.0), "delta_radius == 0");
		runner.expect(near(x(app::auto_aim::kStateDeltaZ), 0.0), "delta_z == 0");

		runner.end();
	}

	// ============================================================
	// Test：initial identity
	// ============================================================

	void test_initial_identity(TestRunner& runner)
	{
		runner.begin("Initial identity");

		auto o = make_observation(0.0, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                          app::auto_aim::ArmorName::Sentry,
		                          app::auto_aim::ArmorType::Small);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
		                                              app::auto_aim::kTargetStateDim);

		app::auto_aim::Target target(o, 0.2, P0, deterministic_config());

		runner.expect(target.color() == app::auto_aim::ArmorColor::Red, "color preserved");
		runner.expect(target.name() == app::auto_aim::ArmorName::Sentry, "name preserved");
		runner.expect(target.type() == app::auto_aim::ArmorType::Small, "type preserved");

		runner.end();
	}

	// ============================================================
	// Test：armor count rule
	// ============================================================

	void test_armor_count_rule(TestRunner& runner)
	{
		runner.begin("Armor count rule");

		using namespace app::auto_aim;

		runner.expect(
		    armor_count_for(ArmorType::Big, ArmorName::Three) == 2,
		    "Big Three -> 2");
		runner.expect(
		    armor_count_for(ArmorType::Big, ArmorName::Four) == 2,
		    "Big Four -> 2");
		runner.expect(
		    armor_count_for(ArmorType::Big, ArmorName::Five) == 2,
		    "Big Five -> 2");
		runner.expect(
		    armor_count_for(ArmorType::Big, ArmorName::Outpost) == 3,
		    "Outpost -> 3 (even if big)");
		runner.expect(
		    armor_count_for(ArmorType::Small, ArmorName::Outpost) == 3,
		    "Outpost -> 3");
		runner.expect(
		    armor_count_for(ArmorType::Big, ArmorName::Base) == 3,
		    "Base -> 3");
		runner.expect(
		    armor_count_for(ArmorType::Small, ArmorName::Base) == 3,
		    "Base -> 3");
		runner.expect(
		    armor_count_for(ArmorType::Small, ArmorName::Four) == 4,
		    "Small Four -> 4 (default)");
		runner.expect(
		    armor_count_for(ArmorType::Small, ArmorName::Sentry) == 4,
		    "Small Sentry -> 4 (default)");
		runner.expect(
		    armor_count_for(ArmorType::Unknown, ArmorName::NotArmor) == 4,
		    "Unknown -> 4 (default)");

		runner.end();
	}

	// ============================================================
	// Test：predict (dt=0, constant velocity, constant yaw rate)
	// ============================================================

	void test_predict(TestRunner& runner)
	{
		runner.begin("Predict");

		auto o = make_observation(0.0, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                          app::auto_aim::ArmorName::Four,
		                          app::auto_aim::ArmorType::Small);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
		                                              app::auto_aim::kTargetStateDim);

		app::auto_aim::Target target(o, 0.2, P0, deterministic_config());
		const Eigen::VectorXd prior = target.state();

		// dt == 0：状态不变。
		target.predict(0.0);
		runner.expect((target.state() - prior).norm() <= 1e-12, "dt==0 keeps state unchanged");

		// 直接构造一个有速度 / yaw_rate 的状态比较麻烦；通过 F 矩阵解析验证运动学。
		// 这里单独验证 F 的精确值。
		runner.end();
	}

	void test_transition_matrix_exact(TestRunner& runner)
	{
		runner.begin("Transition matrix exact");

		using namespace app::auto_aim;

		const double dt = 0.1;
		const Eigen::MatrixXd F = Target::transition_matrix(dt);

		runner.expect(F.rows() == 11 && F.cols() == 11, "F is 11x11");
		runner.expect(near(F(kStateX, kStateVx), dt), "F(x,vx) == dt");
		runner.expect(near(F(kStateY, kStateVy), dt), "F(y,vy) == dt");
		runner.expect(near(F(kStateZ, kStateVz), dt), "F(z,vz) == dt");
		runner.expect(near(F(kStateYaw, kStateYawRate), dt), "F(yaw,yaw_rate) == dt");

		// 其余 off-diagonal 为 0，对角为 1。
		Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(11, 11);
		expected(kStateX, kStateVx) = dt;
		expected(kStateY, kStateVy) = dt;
		expected(kStateZ, kStateVz) = dt;
		expected(kStateYaw, kStateYawRate) = dt;

		runner.expect((F - expected).norm() <= 1e-12, "F matches analytic expectation");

		runner.end();
	}

	void test_constant_velocity_predict(TestRunner& runner)
	{
		runner.begin("Constant velocity predict");

		// center=(0,0,0), 通过构造速度为 (1,2,3)、yaw_rate=0.5 的 state 用 F 直接推进。
		Eigen::VectorXd x = Eigen::VectorXd::Zero(app::auto_aim::kTargetStateDim);
		x(app::auto_aim::kStateVx) = 1.0;
		x(app::auto_aim::kStateVy) = 2.0;
		x(app::auto_aim::kStateVz) = 3.0;
		x(app::auto_aim::kStateYawRate) = 0.5;

		const double dt = 0.2;
		Eigen::MatrixXd F = app::auto_aim::Target::transition_matrix(dt);

		Eigen::VectorXd x_next = F * x;
		x_next(app::auto_aim::kStateYaw) =
		    app::auto_aim::wrap_angle(x_next(app::auto_aim::kStateYaw));

		runner.expect(near(x_next(app::auto_aim::kStateX), 0.2), "x == vx*dt");
		runner.expect(near(x_next(app::auto_aim::kStateY), 0.4), "y == vy*dt");
		runner.expect(near(x_next(app::auto_aim::kStateZ), 0.6), "z == vz*dt");
		runner.expect(near(x_next(app::auto_aim::kStateYaw), 0.1), "yaw == yaw_rate*dt");
		runner.expect(near(x_next(app::auto_aim::kStateVx), 1.0), "vx constant");
		runner.expect(near(x_next(app::auto_aim::kStateYawRate), 0.5), "yaw_rate constant");

		runner.end();
	}

	void test_yaw_wrap_crossing(TestRunner& runner)
	{
		runner.begin("Yaw wrap crossing");

		Eigen::VectorXd x = Eigen::VectorXd::Zero(app::auto_aim::kTargetStateDim);
		x(app::auto_aim::kStateYaw) = kPi - 0.01;
		x(app::auto_aim::kStateYawRate) = 1.0;

		const double dt = 0.05;
		Eigen::MatrixXd F = app::auto_aim::Target::transition_matrix(dt);

		Eigen::VectorXd x_next = F * x;
		x_next(app::auto_aim::kStateYaw) =
		    app::auto_aim::wrap_angle(x_next(app::auto_aim::kStateYaw));

		// pi - 0.01 + 0.05 = pi + 0.04 -> wrap to -pi + 0.04。
		runner.expect(near(x_next(app::auto_aim::kStateYaw), -kPi + 0.04, 1e-12),
		              "yaw wraps across +pi to [-pi, pi)");
		runner.expect(x_next(app::auto_aim::kStateYaw) >= -kPi,
		              "wrapped yaw >= -pi");
		runner.expect(x_next(app::auto_aim::kStateYaw) < kPi, "wrapped yaw < pi");

		runner.end();
	}

	// ============================================================
	// Test：process noise Q
	// ============================================================

	void test_process_noise(TestRunner& runner)
	{
		runner.begin("Process noise");

		const double dt = 0.1;
		const Eigen::MatrixXd Q =
		    app::auto_aim::Target::process_noise_matrix(dt, deterministic_config());

		runner.expect(Q.rows() == 11 && Q.cols() == 11, "Q is 11x11");
		runner.expect(Q.allFinite(), "Q all finite");
		runner.expect((Q - Q.transpose()).norm() <= 1e-12, "Q symmetric");

		const double dt2 = dt * dt;
		const double dt3 = dt2 * dt;
		const double dt4 = dt2 * dt2;

		runner.expect(near(Q(app::auto_aim::kStateX, app::auto_aim::kStateX), dt4 / 4.0),
		              "Q(xx) == dt^4/4");
		runner.expect(near(Q(app::auto_aim::kStateX, app::auto_aim::kStateVx), dt3 / 2.0),
		              "Q(x,vx) == dt^3/2");
		runner.expect(near(Q(app::auto_aim::kStateVx, app::auto_aim::kStateVx), dt2),
		              "Q(vx,vx) == dt^2");
		runner.expect(near(Q(app::auto_aim::kStateRadius, app::auto_aim::kStateRadius), dt),
		              "Q(radius,radius) == dt");

		runner.end();
	}

	// ============================================================
	// Test：geometry (armor hypotheses)
	// ============================================================

	void test_armor_geometry(TestRunner& runner)
	{
		runner.begin("Armor geometry");

		// 2 armor：center (0,0,0), yaw=0, radius=0.2。
		{
			auto o = make_observation(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                          app::auto_aim::ArmorName::Three,
			                          app::auto_aim::ArmorType::Big);
			Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
			                                              app::auto_aim::kTargetStateDim);
			app::auto_aim::Target t(o, 0.2, P0, deterministic_config());

			runner.expect(t.armor_count() == 2, "2-armor target count == 2");

			const auto hyps = t.armor_hypotheses();
			runner.expect(hyps.size() == 2, "2 hypotheses");

			runner.expect(near(hyps[0].position_in_world.x(), -0.2), "armor0 x");
			runner.expect(near(hyps[0].position_in_world.y(), 0.0), "armor0 y");
			runner.expect(near(hyps[1].position_in_world.x(), 0.2), "armor1 x (theta=±pi)");
			runner.expect(near(hyps[1].position_in_world.y(), 0.0, 1e-9),
			              "armor1 y (theta=±pi)");
		}

		// 3 armor：Outpost。
		{
			auto o = make_observation(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                          app::auto_aim::ArmorName::Outpost,
			                          app::auto_aim::ArmorType::Small);
			Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
			                                              app::auto_aim::kTargetStateDim);
			app::auto_aim::Target t(o, 0.2, P0, deterministic_config());

			runner.expect(t.armor_count() == 3, "3-armor target count == 3");
			runner.expect(t.armor_hypotheses().size() == 3, "3 hypotheses");
		}

		// 4 armor default。
		{
			auto o = make_observation(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                          app::auto_aim::ArmorName::Four,
			                          app::auto_aim::ArmorType::Small);
			Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
			                                              app::auto_aim::kTargetStateDim);
			app::auto_aim::Target t(o, 0.2, P0, deterministic_config());

			runner.expect(t.armor_count() == 4, "4-armor target count == 4");
			runner.expect(t.armor_hypotheses().size() == 4, "4 hypotheses");
		}

		runner.end();
	}

	void test_all_hypotheses_finite(TestRunner& runner)
	{
		runner.begin("All hypotheses finite");

		const app::auto_aim::Target t = make_target(0.0, 0.0, 0.0, 0.3, 0.2);

		bool all_finite = true;
		for(const auto& h: t.armor_hypotheses())
		{
			all_finite = all_finite && h.position_in_world.allFinite()
			    && std::isfinite(h.yaw_in_world);
		}

		runner.expect(all_finite, "all hypotheses finite");

		runner.end();
	}

	// ============================================================
	// Test：measurement model exact
	// ============================================================

	void test_measurement_model_exact(TestRunner& runner)
	{
		runner.begin("Measurement model exact");

		// 非边界状态：center (1,2,0.5), yaw=0.3, r=0.2，armor 0。
		app::auto_aim::Target t = make_target(1.0, 2.0, 0.5, 0.3, 0.2);

		const Eigen::VectorXd x = t.state();

		// armor 0：theta=0.3，position = center - r*(cos,sin)。
		const double ax = 1.0 - 0.2 * std::cos(0.3);
		const double ay = 2.0 - 0.2 * std::sin(0.3);
		const double az = 0.5;

		const double expected_yaw = std::atan2(ay, ax);
		const double expected_pitch = std::atan2(az, std::sqrt(ax * ax + ay * ay));
		const double expected_distance = std::sqrt(ax * ax + ay * ay + az * az);

		const Eigen::Vector4d z0 = t.measurement_model(x, 0);
		runner.expect(near(z0.x(), expected_yaw), "armor0 bearing_yaw");
		runner.expect(near(z0.y(), expected_pitch), "armor0 pitch");
		runner.expect(near(z0.z(), expected_distance), "armor0 distance");
		runner.expect(near(z0.w(), 0.3), "armor0 armor_yaw");

		runner.end();
	}

	void test_delta_geometry_exact(TestRunner& runner)
	{
		runner.begin("Delta geometry exact");

		// 非边界状态 + alternate 几何。
		app::auto_aim::Target t = make_target(0.0, 0.0, 0.0, 0.3, 0.2);

		Eigen::VectorXd x = t.state();
		x(app::auto_aim::kStateDeltaRadius) = 0.05;
		x(app::auto_aim::kStateDeltaZ) = 0.1;

		// armor 1 (alternate)：theta = 0.3 + pi/2，r = 0.25，z = 0.1。
		{
			const double theta = 0.3 + kPi / 2.0;
			const double ax = -0.25 * std::cos(theta);
			const double ay = -0.25 * std::sin(theta);
			const double az = 0.1;
			const Eigen::Vector4d z1 = t.measurement_model(x, 1);
			runner.expect(near(z1.x(), std::atan2(ay, ax), 1e-9), "armor1 bearing_yaw");
			runner.expect(near(z1.y(), std::atan2(az, std::sqrt(ax * ax + ay * ay)), 1e-9), "armor1 pitch");
			runner.expect(near(z1.z(), std::sqrt(ax * ax + ay * ay + az * az), 1e-9), "armor1 distance (r+dr)");
			runner.expect(near(z1.w(), theta, 1e-9), "armor1 armor_yaw");
		}

		// armor 0 (non-alternate)：theta = 0.3，r = 0.2，z = 0（不受 delta 影响）。
		{
			const double theta = 0.3;
			const double ax = -0.2 * std::cos(theta);
			const double ay = -0.2 * std::sin(theta);
			const double az = 0.0;
			const Eigen::Vector4d z0 = t.measurement_model(x, 0);
			runner.expect(near(z0.x(), std::atan2(ay, ax), 1e-9), "armor0 bearing_yaw");
			runner.expect(near(z0.y(), std::atan2(az, std::sqrt(ax * ax + ay * ay)), 1e-9), "armor0 pitch");
			runner.expect(near(z0.z(), std::sqrt(ax * ax + ay * ay + az * az), 1e-9), "armor0 distance (r only)");
			runner.expect(near(z0.w(), theta, 1e-9), "armor0 armor_yaw");
		}

		runner.end();
	}

	// ============================================================
	// Test：analytic Jacobian vs finite difference
	// ============================================================

	void test_jacobian_finite_difference(TestRunner& runner)
	{
		runner.begin("Jacobian vs finite difference");

		app::auto_aim::Target t = make_target(1.0, 2.0, 0.5, 0.3, 0.2);

		Eigen::VectorXd x = t.state();
		x(app::auto_aim::kStateVx) = 0.1;
		x(app::auto_aim::kStateVy) = 0.2;
		x(app::auto_aim::kStateYawRate) = 0.3;
		x(app::auto_aim::kStateDeltaRadius) = 0.05;
		x(app::auto_aim::kStateDeltaZ) = 0.1;

		const int armor_id = 1; // alternate board，覆盖 delta_radius/delta_z 导数。

		const Eigen::MatrixXd H = t.measurement_jacobian(x, armor_id);

		const double h = 1e-6;
		const double tol = 1e-6;

		bool all_close = true;
		std::vector<std::string> bad;

		for(int j = 0; j < app::auto_aim::kTargetStateDim; ++j)
		{
			Eigen::VectorXd xp = x;
			Eigen::VectorXd xm = x;

			xp(j) += h;
			xm(j) -= h;

			const Eigen::Vector4d zp = t.measurement_model(xp, armor_id);
			const Eigen::Vector4d zm = t.measurement_model(xm, armor_id);

			Eigen::Vector4d grad;
			grad.x() = tools::maths_tools::limit_rad(zp.x() - zm.x()) / (2.0 * h);
			grad.y() = tools::maths_tools::limit_rad(zp.y() - zm.y()) / (2.0 * h);
			grad.z() = (zp.z() - zm.z()) / (2.0 * h);
			grad.w() = tools::maths_tools::limit_rad(zp.w() - zm.w()) / (2.0 * h);

			const Eigen::Vector4d analytic = H.col(j);

			if((grad - analytic).norm() > tol)
			{
				all_close = false;
				bad.push_back(std::to_string(j));
			}
		}

		runner.expect(all_close,
		              std::string("all Jacobian columns match central difference")
		                  + (bad.empty() ? "" : (" (bad: " + bad.front() + ")")));

		runner.end();
	}

	void test_measurement_model_armor_counts(TestRunner& runner)
	{
		runner.begin("Measurement model armor counts");

		// 3-armor (Outpost)。
		{
			auto o = make_observation(1.0, 1.0, 0.0, 0.3, app::auto_aim::ArmorColor::Red,
			                          app::auto_aim::ArmorName::Outpost,
			                          app::auto_aim::ArmorType::Big);
			Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
			                                              app::auto_aim::kTargetStateDim);
			app::auto_aim::Target t(o, 0.2765, P0, deterministic_config());
			runner.expect(t.armor_count() == 3, "outpost armor_count == 3");
			for(int i = 0; i < t.armor_count(); ++i)
			{
				const Eigen::Vector4d z = t.measurement_model(t.state(), i);
				runner.expect(z.allFinite(), "3-armor measurement finite");
			}
		}

		// 2-armor (Big Three)。
		{
			auto o = make_observation(1.0, 1.0, 0.0, 0.3, app::auto_aim::ArmorColor::Blue,
			                          app::auto_aim::ArmorName::Three,
			                          app::auto_aim::ArmorType::Big);
			Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
			                                              app::auto_aim::kTargetStateDim);
			app::auto_aim::Target t(o, 0.2, P0, deterministic_config());
			runner.expect(t.armor_count() == 2, "big three armor_count == 2");
			for(int i = 0; i < t.armor_count(); ++i)
			{
				const Eigen::Vector4d z = t.measurement_model(t.state(), i);
				runner.expect(z.allFinite(), "2-armor measurement finite");
			}
		}

		runner.end();
	}

	void test_measurement_model_armor_yaw_boundary(TestRunner& runner)
	{
		runner.begin("Measurement model armor_yaw ±pi boundary");

		// 4-armor, yaw=0, armor 2 -> theta=pi -> limit_rad -> +pi。
		app::auto_aim::Target t = make_target(0.0, 0.0, 0.0, 0.0, 0.2);
		const Eigen::Vector4d z2 = t.measurement_model(t.state(), 2);
		runner.expect(near(z2.w(), kPi, 1e-9), "armor2 armor_yaw == +pi (limit_rad)");

		runner.end();
	}

	void test_measurement_residual_direct(TestRunner& runner)
	{
		runner.begin("Measurement residual direct (wrap)");

		Eigen::VectorXd z(4);
		Eigen::VectorXd h(4);

		// ============================================================
		// 判别性用例：raw residual 恰好 = ±pi。
		// limit_rad((-pi, pi]) 返回 +pi；若误用 wrap_angle([-pi, pi)) 会返回 -pi，
		// 用于杀死 limit_rad -> wrap_angle mutant。
		// ============================================================

		// index 0 (bearing_yaw)。
		z << kPi, 0.0, 1.0, 0.0;
		h << 0.0, 0.0, 1.0, 0.0;
		Eigen::VectorXd r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(0), kPi, 1e-12), "bearing_yaw raw +pi -> +pi");

		z << -kPi, 0.0, 1.0, 0.0;
		h << 0.0, 0.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(0), kPi, 1e-12), "bearing_yaw raw -pi -> +pi");

		// index 1 (pitch)。
		z << 0.0, kPi, 1.0, 0.0;
		h << 0.0, 0.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(1), kPi, 1e-12), "pitch raw +pi -> +pi");

		z << 0.0, -kPi, 1.0, 0.0;
		h << 0.0, 0.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(1), kPi, 1e-12), "pitch raw -pi -> +pi");

		// index 3 (armor_yaw)。
		z << 0.0, 0.0, 1.0, kPi;
		h << 0.0, 0.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(3), kPi, 1e-12), "armor_yaw raw +pi -> +pi");

		z << 0.0, 0.0, 1.0, -kPi;
		h << 0.0, 0.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(3), kPi, 1e-12), "armor_yaw raw -pi -> +pi");

		// ============================================================
		// 非判别 sanity：raw residual = ±2pi -> 0（两种 convention 相同）。
		// ============================================================
		z << kPi, 0.0, 5.0, 0.0;
		h << -kPi, 0.0, 3.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(0), 0.0, 1e-12), "bearing_yaw 2pi -> 0 (sanity)");
		runner.expect(near(r(2), 2.0, 1e-12), "distance not wrapped (5-3=2)");

		// index 1 大角度 wrap（raw 6.0 -> 6.0-2pi）。
		z << 0.0, 3.0, 1.0, 0.0;
		h << 0.0, -3.0, 1.0, 0.0;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(1), 6.0 - kTwoPi, 1e-9), "pitch 6.0 -> 6.0-2pi (sanity)");

		// index 3 2pi -> 0。
		z << 0.0, 0.0, 1.0, kPi;
		h << 0.0, 0.0, 1.0, -kPi;
		r = app::auto_aim::Target::measurement_residual(z, h);
		runner.expect(near(r(3), 0.0, 1e-12), "armor_yaw 2pi -> 0 (sanity)");

		// limit_rad 函数本身的 convention。
		runner.expect(near(tools::maths_tools::limit_rad(kPi), kPi, 1e-12), "limit_rad(+pi) == +pi");
		runner.expect(near(tools::maths_tools::limit_rad(-kPi), kPi, 1e-12), "limit_rad(-pi) == +pi");

		runner.end();
	}

	void test_measurement_covariance_direct(TestRunner& runner)
	{
		runner.begin("Measurement covariance direct (adaptive R)");

		app::auto_aim::MeasurementNoiseConfig config;
		config.base_covariance = Eigen::MatrixXd::Zero(4, 4);
		config.base_covariance(0, 0) = 4e-3;
		config.base_covariance(1, 1) = 4e-3;
		config.base_covariance(2, 2) = 1.0;
		config.base_covariance(3, 3) = 9e-2;
		config.distance_angle_log_gain = 1.0;
		config.armor_yaw_distance_log_gain = 1.0 / 200.0;

		// 观测：position (1,1,0)，armor_yaw=0.5，distance=sqrt(2)。
		app::auto_aim::ArmorObservation obs;
		obs.position_in_world = Eigen::Vector3d(1.0, 1.0, 0.0);
		obs.ypd_in_world = Eigen::Vector3d(kPi / 4.0, 0.0, std::sqrt(2.0));
		obs.armor_yaw_in_world = 0.5;

		const Eigen::MatrixXd R = app::auto_aim::Target::measurement_covariance(obs, config);

		const double center_yaw = std::atan2(obs.position_in_world.y(), obs.position_in_world.x());
		const double delta_angle = tools::maths_tools::limit_rad(obs.armor_yaw_in_world - center_yaw);
		const double distance = obs.ypd_in_world.z();

		const double expected_r22 = 1.0 + 1.0 * std::log1p(std::abs(delta_angle));
		const double expected_r33 = 9e-2 + (1.0 / 200.0) * std::log1p(std::abs(distance));

		runner.expect(near(R(0, 0), 4e-3), "R00 constant 4e-3");
		runner.expect(near(R(1, 1), 4e-3), "R11 constant 4e-3");
		runner.expect(near(R(2, 2), expected_r22, 1e-12), "R22 vs delta_angle");
		runner.expect(near(R(3, 3), expected_r33, 1e-12), "R33 vs distance");
		runner.expect(near(R(0, 1), 0.0) && near(R(1, 0), 0.0) && near(R(0, 2), 0.0)
		                  && near(R(0, 3), 0.0) && near(R(1, 2), 0.0) && near(R(1, 3), 0.0)
		                  && near(R(2, 3), 0.0) && near(R(3, 2), 0.0),
		              "R is diagonal");

		// gain=0 -> R == base。
		config.distance_angle_log_gain = 0.0;
		config.armor_yaw_distance_log_gain = 0.0;
		const Eigen::MatrixXd R0 = app::auto_aim::Target::measurement_covariance(obs, config);
		runner.expect(near(R0(2, 2), 1.0), "R22 == base when gain 0");
		runner.expect(near(R0(3, 3), 9e-2), "R33 == base when gain 0");

		runner.end();
	}

	// ============================================================
	// Test：public helper contract（P2-3）
	// ============================================================

	void test_helper_contracts(TestRunner& runner)
	{
		runner.begin("Public helper contracts");

		app::auto_aim::Target t = make_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// 正确调用不应抛异常。
		{
			bool threw = false;

			try
			{
				(void)t.measurement_model(t.state(), 0);
				(void)t.measurement_jacobian(t.state(), 0);
			}
			catch(...)
			{
				threw = true;
			}

			runner.expect(!threw, "valid inputs do not throw");
		}

		// x 维度错误。
		{
			bool threw = false;
			Eigen::VectorXd bad(3);
			bad.setZero();

			try
			{
				(void)t.measurement_model(bad, 0);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "wrong x dimension throws invalid_argument");
		}

		// x 含 NaN。
		{
			bool threw = false;
			Eigen::VectorXd bad = t.state();
			bad(app::auto_aim::kStateX) = std::numeric_limits<double>::quiet_NaN();

			try
			{
				(void)t.measurement_jacobian(bad, 0);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "non-finite x throws invalid_argument");
		}

		// armor_id 越界。
		{
			bool threw = false;

			try
			{
				(void)t.measurement_model(t.state(), 99);
			}
			catch(const std::invalid_argument&)
			{
				threw = true;
			}

			runner.expect(threw, "armor_id out of range throws invalid_argument");
		}

		runner.end();
	}

	// ============================================================
	// Test：only Eigen / tracker types (no Detector/Solver dependency)
	// ============================================================

	void test_no_detector_solver_dependency(TestRunner& runner)
	{
		runner.begin("No Detector/Solver dependency");

		// target.hpp / target.cpp 只依赖 Eigen 与 Tracker 边界类型。
		// 本测试能独立编译并链接 app 库中的 target.cpp 即证明不拉入
		// Detector/Solver/OpenVINO 运行时依赖（若存在，此处会链接失败）。
		runner.expect(true, "target compiles against Eigen + tracker types only");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Target Module Test Suite ===\n\n");

	TestRunner runner;

	test_state_dimension(runner);
	test_initialization_geometry(runner);
	test_initial_identity(runner);
	test_armor_count_rule(runner);
	test_predict(runner);
	test_transition_matrix_exact(runner);
	test_constant_velocity_predict(runner);
	test_yaw_wrap_crossing(runner);
	test_process_noise(runner);
	test_armor_geometry(runner);
	test_all_hypotheses_finite(runner);
	test_measurement_model_exact(runner);
	test_delta_geometry_exact(runner);
	test_jacobian_finite_difference(runner);
	test_measurement_model_armor_counts(runner);
	test_measurement_model_armor_yaw_boundary(runner);
	test_measurement_residual_direct(runner);
	test_measurement_covariance_direct(runner);
	test_helper_contracts(runner);
	test_no_detector_solver_dependency(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Target tests failed ===\n");
		return 1;
	}

	std::printf("=== All target tests passed ===\n");
	return 0;
}