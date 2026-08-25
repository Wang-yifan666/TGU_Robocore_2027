/**
 * @file test_association.cpp
 * @brief Association + Target correction 确定性单元测试。
 *
 * 无相机 / 无 OpenVINO / 无视频 / 无随机依赖。
 */

#include "app/auto_aim/association.hpp"
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

	app::auto_aim::TargetModelConfig model_config()
	{
		app::auto_aim::TargetModelConfig c;
		c.translation_accel_variance = 1.0;
		c.yaw_accel_variance = 1.0;
		c.radius_random_walk_variance = 1.0;
		c.delta_radius_random_walk_variance = 1.0;
		c.delta_z_random_walk_variance = 1.0;
		return c;
	}

	app::auto_aim::AssociationConfig assoc_config()
	{
		app::auto_aim::AssociationConfig c;
		c.max_position_error_m = 0.5;
		c.max_yaw_error_rad = 1.0;
		c.position_score_scale_m = 1.0;
		c.yaw_score_scale_rad = 1.0;
		return c;
	}

	app::auto_aim::ArmorObservation make_obs(double x, double y, double z, double yaw,
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
		o.ypd_in_world = tools::maths_tools::xyz2ypd(Eigen::Vector3d(x, y, z));
		return o;
	}

	// 构造 4-armor 车辆 target，center + yaw + radius，P0=identity。
	app::auto_aim::Target build_target(double cx, double cy, double cz, double yaw, double radius)
	{
		const double obs_x = cx - radius * std::cos(yaw);
		const double obs_y = cy - radius * std::sin(yaw);

		auto o = make_obs(obs_x, obs_y, cz, yaw, app::auto_aim::ArmorColor::Red,
		                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(app::auto_aim::kTargetStateDim,
		                                              app::auto_aim::kTargetStateDim);

		return app::auto_aim::Target(o, radius, P0, model_config());
	}

	app::auto_aim::MeasurementNoiseConfig measurement_noise(double sigma2)
	{
		app::auto_aim::MeasurementNoiseConfig config;
		config.base_covariance = sigma2 * Eigen::MatrixXd::Identity(
		    app::auto_aim::kTargetMeasurementDim, app::auto_aim::kTargetMeasurementDim);
		config.distance_angle_log_gain = 0.0;
		config.armor_yaw_distance_log_gain = 0.0;
		return config;
	}

	// ============================================================
	// Test：single obvious association
	// ============================================================

	void test_single_obvious(TestRunner& runner)
	{
		runner.begin("Single obvious association");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// armor 0 在 (-0.2, 0, 0)。
		std::vector<app::auto_aim::ArmorObservation> obs = {
		    make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small)};

		const auto result = app::auto_aim::associate(target, obs, assoc_config());

		runner.expect(result.has_value(), "association found");
		runner.expect(result && result->observation_index == 0, "observation index == 0");
		runner.expect(result && result->armor_id == 0, "armor_id == 0");

		runner.end();
	}

	// ============================================================
	// Test：correct armor hypothesis selection across multiple hypotheses
	// ============================================================

	void test_multiple_hypotheses(TestRunner& runner)
	{
		runner.begin("Multiple hypotheses");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// observation 在 armor 2 位置 (0.2, 0, 0), yaw = pi。
		std::vector<app::auto_aim::ArmorObservation> obs = {
		    make_obs(0.2, 0.0, 0.0, kPi, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small)};

		const auto result = app::auto_aim::associate(target, obs, assoc_config());

		runner.expect(result.has_value(), "association found");
		runner.expect(result && result->armor_id == 2, "matched armor_id == 2");

		runner.end();
	}

	// ============================================================
	// Test：multiple observations
	// ============================================================

	void test_multiple_observations(TestRunner& runner)
	{
		runner.begin("Multiple observations");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// obs[0] = 远距离 outlier（应被 position gate 拒绝）。
		// obs[1] = 正确 armor 2 观测（yaw 用 -pi，与 hypothesis yaw 精确一致）。
		std::vector<app::auto_aim::ArmorObservation> obs = {
		    make_obs(10.0, 10.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small),
		    make_obs(0.2, 0.0, 0.0, -kPi, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small)};

		const auto result = app::auto_aim::associate(target, obs, assoc_config());

		runner.expect(result.has_value(), "association found");
		runner.expect(result && result->observation_index == 1,
		              "outlier rejected, correct observation selected");
		runner.expect(result && result->armor_id == 2, "matched armor 2");

		runner.end();
	}

	// ============================================================
	// Test：identity rejection (name / type / color)
	// ============================================================

	void test_identity_rejection(TestRunner& runner)
	{
		runner.begin("Identity rejection");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// wrong name。
		{
			auto obs = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                    app::auto_aim::ArmorName::Three,
			                    app::auto_aim::ArmorType::Small);
			runner.expect(!app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "wrong name rejected");
		}

		// wrong type。
		{
			auto obs = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Big);
			runner.expect(!app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "wrong type rejected");
		}

		// color mismatch (both not Unknown)。
		{
			auto obs = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Blue,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Small);
			runner.expect(!app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "color mismatch rejected");
		}

		// Unknown color compatibility。
		{
			auto obs = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Unknown,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Small);
			runner.expect(app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "Unknown observation color accepted");
		}

		runner.end();
	}

	// ============================================================
	// Test：position / yaw hard gate
	// ============================================================

	void test_gates(TestRunner& runner)
	{
		runner.begin("Position / yaw gate");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// position gate：超出 max_position_error。
		{
			auto obs = make_obs(10.0, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Small);
			runner.expect(!app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "position gate rejects far observation");
		}

		// yaw gate：yaw 相差过大（target armor 0 yaw=0, obs yaw=0.5 rad）。
		{
			auto cfg = assoc_config();
			cfg.max_yaw_error_rad = 0.1;

			auto obs = make_obs(-0.2, 0.0, 0.0, 0.5, app::auto_aim::ArmorColor::Red,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Small);
			runner.expect(!app::auto_aim::associate(target, {obs}, cfg).has_value(),
			              "yaw gate rejects far yaw");
		}

		runner.end();
	}

	// ============================================================
	// Test：±pi wrap in association
	// ============================================================

	void test_yaw_wrap(TestRunner& runner)
	{
		runner.begin("Yaw wrap in association");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// armor 0 yaw = 0，observation yaw = 2pi。raw subtraction 会是 2pi，
		// 但 wrap 后应为 0；用很小 yaw gate 验证 wrap 生效。
		auto cfg = assoc_config();
		cfg.max_yaw_error_rad = 0.1;

		auto obs = make_obs(-0.2, 0.0, 0.0, 2.0 * kPi, app::auto_aim::ArmorColor::Red,
		                    app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		const auto result = app::auto_aim::associate(target, {obs}, cfg);

		runner.expect(result.has_value(), "2pi-offset yaw still associates (wrapped)");
		runner.expect(result && near(result->yaw_residual, 0.0, 1e-9),
		              "wrapped yaw residual ≈ 0");

		runner.end();
	}

	// ============================================================
	// Test：deterministic tie breaking
	// ============================================================

	void test_deterministic_tie(TestRunner& runner)
	{
		runner.begin("Deterministic tie");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// 两个完全相同的观测都精确匹配 armor 0（theta=0，坐标精确），
		// score 均为 0；tie 应按 observation index 小者胜。
		std::vector<app::auto_aim::ArmorObservation> obs = {
		    make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small),
		    make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		             app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small)};

		const auto result = app::auto_aim::associate(target, obs, assoc_config());

		runner.expect(result.has_value(), "association found");
		runner.expect(result && result->observation_index == 0,
		              "deterministic tie chooses lower observation index");
		runner.expect(result && result->armor_id == 0, "matched armor 0");

		runner.end();
	}

	// ============================================================
	// Test：no valid pair
	// ============================================================

	void test_no_valid_pair(TestRunner& runner)
	{
		runner.begin("No valid pair");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		runner.expect(!app::auto_aim::associate(target, {}, assoc_config()).has_value(),
		              "empty observations -> nullopt");

		// 非 finite observation 被跳过。
		{
			auto obs = make_obs(0.0, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
			                    app::auto_aim::ArmorName::Four,
			                    app::auto_aim::ArmorType::Small);
			obs.position_in_world.x() = std::numeric_limits<double>::quiet_NaN();
			runner.expect(!app::auto_aim::associate(target, {obs}, assoc_config()).has_value(),
			              "non-finite observation -> nullopt");
		}

		runner.end();
	}

	// ============================================================
	// Test：correction exact / simple case
	// ============================================================

	void test_correction_exact(TestRunner& runner)
	{
		runner.begin("Correction exact");

		app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// armor 0，观测 x 偏移 +0.02（-0.18 vs -0.2），yaw=0。
		auto obs = make_obs(-0.18, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                    app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		const bool ok = target.correct(obs, 0, measurement_noise(1e-4));

		runner.expect(ok, "correction returns true");
		runner.expect(target.last_innovation().size() == 4, "innovation is 4D");
		runner.expect(near(target.last_innovation()(2), -0.02, 1e-6),
		              "innovation distance ≈ -0.02");
		runner.expect(std::isfinite(target.last_nis()), "NIS finite");
		runner.expect(target.state().allFinite(), "state finite after correction");

		runner.end();
	}

	// ============================================================
	// Test：correction false leaves prior unchanged
	// ============================================================

	void test_correction_false_rollback(TestRunner& runner)
	{
		runner.begin("Correction false rollback");

		// 用 zero P0 和 zero R 构造奇异 S，使 update 返回 false。
		auto o = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		Eigen::MatrixXd P0 =
		    Eigen::MatrixXd::Zero(app::auto_aim::kTargetStateDim, app::auto_aim::kTargetStateDim);

		app::auto_aim::Target target(o, 0.2, P0, model_config());

		const Eigen::VectorXd prior_state = target.state();
		const Eigen::MatrixXd prior_cov = target.covariance();

		auto observation = make_obs(-0.1, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                            app::auto_aim::ArmorName::Four,
		                            app::auto_aim::ArmorType::Small);

		app::auto_aim::MeasurementNoiseConfig zero_noise;
		zero_noise.base_covariance = Eigen::MatrixXd::Zero(4, 4);
		zero_noise.distance_angle_log_gain = 0.0;
		zero_noise.armor_yaw_distance_log_gain = 0.0;
		const bool ok = target.correct(observation, 0, zero_noise);

		runner.expect(!ok, "correction returns false on singular S");
		runner.expect((target.state() - prior_state).norm() <= 1e-12,
		              "state unchanged on failed correction");
		runner.expect((target.covariance() - prior_cov).norm() <= 1e-12,
		              "covariance unchanged on failed correction");

		runner.end();
	}

	// ============================================================
	// Test：vehicle center continuity across armor switch
	// ============================================================

	void test_center_continuity(TestRunner& runner)
	{
		runner.begin("Center continuity across armor switch");

		app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// 先看 armor 0。
		auto obs0 = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                     app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
		runner.expect(target.correct(obs0, 0, measurement_noise(1e-4)),
		              "correction armor0 succeeds");

		const Eigen::VectorXd center_before = target.state().head<3>();

		// 切换到对侧 armor 2 (0.2, 0, 0, yaw=pi)。
		auto obs2 = make_obs(0.2, 0.0, 0.0, kPi, app::auto_aim::ArmorColor::Red,
		                     app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
		runner.expect(target.correct(obs2, 2, measurement_noise(1e-4)),
		              "correction armor2 succeeds");

		const Eigen::VectorXd center_after = target.state().head<3>();

		// 车辆中心不应从 (0,0,0) 跳去 armor 2 一侧（0.4m jump）。
		runner.expect((center_after - center_before).norm() < 0.05,
		              "vehicle center does not jump across armor switch");

		runner.end();
	}

	// ============================================================
	// Test：observation order does not change armor geometry IDs
	// ============================================================

	void test_observation_order_invariant(TestRunner& runner)
	{
		runner.begin("Observation order vs armor id");

		const app::auto_aim::Target target = build_target(0.0, 0.0, 0.0, 0.0, 0.2);

		// 单观测：几何位置 (-0.2,0,0), yaw=0 必须匹配 armor_id 0。
		auto obs0 = make_obs(-0.2, 0.0, 0.0, 0.0, app::auto_aim::ArmorColor::Red,
		                     app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		// 单观测：几何位置 (0.2,0,0), yaw=-pi 必须匹配 armor_id 2。
		auto obs2 = make_obs(0.2, 0.0, 0.0, -kPi, app::auto_aim::ArmorColor::Red,
		                     app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

		// 单测：隔离验证几何 -> armor_id 的映射是固定的。
		{
			auto r = app::auto_aim::associate(target, {obs0}, assoc_config());
			runner.expect(r && r->observation_index == 0 && r->armor_id == 0,
			              "isolated obs0 maps to armor_id 0");
		}
		{
			auto r = app::auto_aim::associate(target, {obs2}, assoc_config());
			runner.expect(r && r->observation_index == 0 && r->armor_id == 2,
			              "isolated obs2 maps to armor_id 2");
		}

		// 重排验证：用 identity 不匹配的 distractor 占位，
		// 验证 obs2 无论位于 index 0 还是 index 1，armor_id 恒为 2
		// （避免浮点 tie 干扰）。
		auto distractor = make_obs(0.2, 0.0, 0.0, -kPi, app::auto_aim::ArmorColor::Red,
		                           app::auto_aim::ArmorName::Three,
		                           app::auto_aim::ArmorType::Small);

		{
			auto r = app::auto_aim::associate(target, {obs2, distractor}, assoc_config());
			runner.expect(r && r->observation_index == 0 && r->armor_id == 2,
			              "obs2 at index 0 maps to armor_id 2");
		}
		{
			auto r = app::auto_aim::associate(target, {distractor, obs2}, assoc_config());
			runner.expect(r && r->observation_index == 1 && r->armor_id == 2,
			              "obs2 at index 1 still maps to armor_id 2");
		}

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Association + Correction Test Suite ===\n\n");

	TestRunner runner;

	test_single_obvious(runner);
	test_multiple_hypotheses(runner);
	test_multiple_observations(runner);
	test_identity_rejection(runner);
	test_gates(runner);
	test_yaw_wrap(runner);
	test_deterministic_tie(runner);
	test_no_valid_pair(runner);
	test_correction_exact(runner);
	test_correction_false_rollback(runner);
	test_center_continuity(runner);
	test_observation_order_invariant(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Association tests failed ===\n");
		return 1;
	}

	std::printf("=== All association tests passed ===\n");
	return 0;
}