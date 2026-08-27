/**
 * @file test_aimer.cpp
 * @brief Aimer 单元测试（纯 synthetic TrackedTarget）。
 *
 * 覆盖：bullet speed、signed delay、gyro 兼容双模式、has_armor_switch、
 * armor lock 生命周期、fixed-point 收敛、deterministic 时间。
 */

#include "app/auto_aim/aimer.hpp"
#include "app/auto_aim/vehicle_prediction.hpp"
#include "tools/maths_tools.hpp"

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

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	bool vector_near(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs, double eps = 1e-9)
	{
		return (lhs - rhs).norm() <= eps;
	}

	auto_aim::AimerConfig make_predictive_config(double hysteresis_rad, double max_advance_s)
	{
		auto_aim::AimerConfig c = auto_aim::make_default_aimer_config();
		c.armor_switch_strategy = auto_aim::ArmorSwitchStrategy::PredictiveHysteresis;
		c.predictive_switch_hysteresis_rad = hysteresis_rad;
		c.predictive_switch_max_advance_s = max_advance_s;
		return c;
	}

	// ============================================================
	// 构造 helper
	// ============================================================

	// 构造一个 Tracking 状态的 TrackedTarget 快照。
	// predicted_armors 只用于提供 armor_count 与 finite 校验占位。
	auto_aim::TrackedTarget make_target(Eigen::Vector3d center, Eigen::Vector3d velocity,
	                                    double yaw, double yaw_rate, double radius,
	                                    double delta_radius, double delta_z, int armor_count,
	                                    auto_aim::ArmorName name = auto_aim::ArmorName::Four,
	                                    double timestamp_s = 0.0, bool has_armor_switch = false,
	                                    std::uint64_t token = 1)
	{
		auto_aim::TrackedTarget t;
		t.state = auto_aim::TrackerState::Tracking;
		t.timestamp_s = timestamp_s;
		t.name = name;
		t.center_in_world = center;
		t.velocity_in_world = velocity;
		t.yaw = yaw;
		t.yaw_rate = yaw_rate;
		t.radius = radius;
		t.delta_radius = delta_radius;
		t.delta_z = delta_z;
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

	// ============================================================
	// 测试用例
	// ============================================================

	void test_bullet_speed_fallback(TestRunner& runner)
	{
		runner.begin("Bullet speed fallback");

		const auto config = auto_aim::make_default_aimer_config(); // policy = fallback
		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 0.0, 0.0, 4);

		const double nan = std::numeric_limits<double>::quiet_NaN();
		const double inf = std::numeric_limits<double>::infinity();

		auto_aim::Aimer a23(config);
		auto_aim::Aimer a10(config);
		auto_aim::Aimer aneg(config);
		auto_aim::Aimer anan(config);
		auto_aim::Aimer ainf(config);

		const auto r23 = a23.aim(target, 1.0, 23.0);
		const auto r10 = a10.aim(target, 1.0, 10.0); // < 14 -> fallback 23
		const auto rneg = aneg.aim(target, 1.0, -5.0);
		const auto rnan = anan.aim(target, 1.0, nan);
		const auto rinf = ainf.aim(target, 1.0, inf);

		runner.expect(r23.valid, "23 valid");
		runner.expect(r10.valid, "10 fallback valid");
		runner.expect(rneg.valid, "negative fallback valid");
		runner.expect(rnan.valid, "NaN fallback valid");
		runner.expect(rinf.valid, "Inf fallback valid");

		runner.expect(near(r10.yaw_rad, r23.yaw_rad), "10 fallback yaw == 23 yaw");
		runner.expect(near(r10.pitch_rad, r23.pitch_rad), "10 fallback pitch == 23 pitch");
		runner.expect(near(rneg.yaw_rad, r23.yaw_rad), "negative fallback yaw == 23 yaw");
		runner.expect(near(rnan.yaw_rad, r23.yaw_rad), "NaN fallback yaw == 23 yaw");
		runner.expect(near(rinf.yaw_rad, r23.yaw_rad), "Inf fallback yaw == 23 yaw");


		runner.end();
	}

	void test_bullet_speed_fail_safe(TestRunner& runner)
	{
		runner.begin("Bullet speed fail_safe");

		auto config = auto_aim::make_default_aimer_config();
		config.invalid_bullet_speed_policy = auto_aim::InvalidBulletSpeedPolicy::FailSafe;

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 0.0, 0.0, 4);

		auto_aim::Aimer aimer(config);
		const auto r = aimer.aim(target, 1.0, 10.0);

		runner.expect(!r.valid, "fail_safe invalid bullet -> not valid");
		runner.expect(r.status == auto_aim::AimStatus::InvalidBulletSpeed, "status InvalidBulletSpeed");

		runner.end();
	}

	void test_delay_signed(TestRunner& runner)
	{
		runner.begin("Delay signed comparison");

		auto config = auto_aim::make_default_aimer_config();
		config.high_speed_delay_s = 0.05;
		config.low_speed_delay_s = 0.01;
		config.decision_speed_rad_s = 10.0;

		auto_aim::Aimer aimer(config);

		auto run = [&](double yaw_rate) {
			const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
			                                0.0, yaw_rate, 0.2, 0.0, 0.0, 4);

			auto_aim::AimerDebugData debug;
			const auto r = aimer.aim(target, 1.0, 23.0, &debug);
			runner.expect(r.valid, "aim should be valid for delay test");
			return debug.t_muzzle_s;
		};

		runner.expect(near(run(20.0), 1.05), "large positive yaw_rate -> high delay");
		runner.expect(near(run(1.0), 1.01), "small positive yaw_rate -> low delay");
		runner.expect(near(run(0.0), 1.01), "zero yaw_rate -> low delay");
		runner.expect(near(run(-20.0), 1.01), "negative yaw_rate -> low delay (signed, not abs)");

		runner.end();
	}

	void test_gyro_compatibility(TestRunner& runner)
	{
		runner.begin("Gyro compatibility (radius vs yaw_rate)");

		// 静止目标（yaw_rate=0，避免旋转影响预测几何）。
		const auto make = []() {
			return make_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.3, 0.0,
			                   0.2, 0.0, 0.0, 4, auto_aim::ArmorName::Four, 0.0, true, 1);
		};

		// 1. 默认 radius mode（radius_threshold=2.0）：radius=0.2 <= 2 -> 非陀螺 -> id0。
		auto default_config = auto_aim::make_default_aimer_config();
		auto_aim::Aimer default_aimer(default_config);
		const auto r_default = default_aimer.aim(make(), 1.0, 23.0);
		runner.expect(r_default.valid, "default radius mode: valid");
		runner.expect(r_default.selected_armor_id == 0, "default radius mode: id 0 (non-gyro)");

		// 2. radius mode，radius_threshold=0.05：radius=0.2 > 0.05 -> 陀螺 -> NoValidArmor。
		auto custom = auto_aim::make_default_aimer_config();
		custom.shootable_angle_threshold_rad = 60.0 / 57.3;
		custom.coming_angle_rad = 20.0 / 57.3;
		custom.leaving_angle_rad = 10.0 / 57.3;
		custom.non_gyro_radius_threshold_m = 0.05;
		custom.non_gyro_yaw_rate_threshold_rad_s = 2.0;

		custom.use_radius_for_gyro_detection = true;
		auto_aim::Aimer radius_gyro(custom);
		const auto r_radius = radius_gyro.aim(make(), 1.0, 23.0);
		runner.expect(!r_radius.valid, "radius mode (radius>threshold): NoValidArmor");
		runner.expect(r_radius.status == auto_aim::AimStatus::NoValidArmor, "status NoValidArmor");

		// 3. yaw_rate mode：yaw_rate=0 <= 2 -> 非陀螺 -> id0（忽略 radius=0.2）。
		custom.use_radius_for_gyro_detection = false;
		auto_aim::Aimer yaw_aimer(custom);
		const auto r_yaw = yaw_aimer.aim(make(), 1.0, 23.0);
		runner.expect(r_yaw.valid, "yaw_rate mode: valid");
		runner.expect(r_yaw.selected_armor_id == 0, "yaw_rate mode: id 0 (non-gyro, ignores radius)");

		runner.end();
	}

	void test_has_armor_switch(TestRunner& runner)
	{
		runner.begin("has_armor_switch");

		auto config = auto_aim::make_default_aimer_config(); // shootable=60/57.3，非陀螺。
		auto_aim::Aimer aimer(config);

		// has_armor_switch == false -> 恒选 id0。
		const auto t_false = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                 0.0, 0.0, 0.2, 0.0, 0.0, 4,
		                                 auto_aim::ArmorName::Four, 0.0, false, 1);
		const auto r_false = aimer.aim(t_false, 1.0, 23.0);
		runner.expect(r_false.valid && r_false.selected_armor_id == 0, "no switch -> id 0");

		// has_armor_switch == true + center_yaw=π/4 -> 两块在 60° 内 -> 锁到 id1（tie）。
		const double r = 3.0;
		const Eigen::Vector3d center(r * std::cos(kPi / 4.0), r * std::sin(kPi / 4.0), 0.0);
		const auto t_true = make_target(center, Eigen::Vector3d::Zero(), 0.0, 0.0, 0.2, 0.0, 0.0,
		                                4, auto_aim::ArmorName::Four, 0.0, true, 1);
		const auto r_true = aimer.aim(t_true, 1.0, 23.0);
		runner.expect(r_true.valid, "switch -> valid");
		runner.expect(r_true.selected_armor_id == 1, "switch -> locked to id 1 (tie)");

		runner.end();
	}

	void test_lock_sequence(TestRunner& runner)
	{
		runner.begin("Lock sequence (stateful)");

		auto config = auto_aim::make_default_aimer_config();
		auto_aim::Aimer aimer(config);

		const double r = 3.0;
		const auto make_center = [&](double theta) {
			return Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), 0.0);
		};

		// 1. center_yaw = π/4（tie）-> lock id1。
		const auto t1 = make_target(make_center(kPi / 4.0), Eigen::Vector3d::Zero(), 0.0, 0.0,
		                            0.2, 0.0, 0.0, 4, auto_aim::ArmorName::Four, 0.0, true, 1);
		const auto r1 = aimer.aim(t1, 1.0, 23.0);
		runner.expect(r1.selected_armor_id == 1, "tie -> lock id 1");

		// 2. center_yaw 略减小 -> 无 lock 时应选 id0，但 lock 保持 id1。
		const auto t2 = make_target(make_center(kPi / 4.0 - 0.05), Eigen::Vector3d::Zero(), 0.0,
		                            0.0, 0.2, 0.0, 0.0, 4, auto_aim::ArmorName::Four, 0.0, true, 1);
		const auto r2 = aimer.aim(t2, 1.0, 23.0);
		runner.expect(r2.selected_armor_id == 1, "lock keeps id 1 after geometry shift");

		// 3. fresh aimer（无 lock）在 t2 上 -> id0（证明 lock 起了作用）。
		auto_aim::Aimer fresh(config);
		const auto r_fresh = fresh.aim(t2, 1.0, 23.0);
		runner.expect(r_fresh.selected_armor_id == 0, "fresh aimer picks id 0");

		// 4. token 变化 -> lock 清空 -> id0。
		auto t3 = t2;
		t3.target_token = 2;
		const auto r3 = aimer.aim(t3, 1.0, 23.0);
		runner.expect(r3.selected_armor_id == 0, "token change resets lock -> id 0");

		// 5. reset -> id0。
		aimer.reset();
		const auto r4 = aimer.aim(t2, 1.0, 23.0);
		runner.expect(r4.selected_armor_id == 0, "reset clears lock -> id 0");

		runner.end();
	}

	void test_fixed_point_deterministic(TestRunner& runner)
	{
		runner.begin("Fixed-point convergence + deterministic");

		auto config = auto_aim::make_default_aimer_config();

		// 静止目标：第一轮 refinement 即收敛。
		const auto stationary = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                    0.0, 0.0, 0.2, 0.0, 0.0, 4);
		auto_aim::Aimer a0(config);
		auto_aim::AimerDebugData d0;
		const auto r0 = a0.aim(stationary, 1.0, 23.0, &d0);
		runner.expect(r0.valid && d0.ballistic_converged, "stationary converges");
		runner.expect(d0.refinement_iterations == 1, "stationary converges in 1 refinement");

		// 运动目标：velocity=(1,0,0), yaw_rate=1.0。
		const auto moving = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
		                                0.0, 1.0, 0.2, 0.0, 0.0, 4);

		auto_aim::Aimer a1(config);
		auto_aim::Aimer a2(config);
		auto_aim::AimerDebugData d1, d2;
		const auto r1 = a1.aim(moving, 1.0, 23.0, &d1);
		const auto r2 = a2.aim(moving, 1.0, 23.0, &d2);

		runner.expect(r1.valid, "moving target valid");
		runner.expect(d1.ballistic_arrival_time_s > d1.t_muzzle_s,
		              "ballistic_arrival > t_muzzle (positive flight time)");
		runner.expect(d1.refinement_iterations <= config.max_refinement_iterations,
		              "iterations <= max");

		// deterministic：两个全新 Aimer，相同输入 -> 逐位一致。
		runner.expect(r1.yaw_rad == r2.yaw_rad, "yaw deterministic");
		runner.expect(r1.pitch_rad == r2.pitch_rad, "pitch deterministic");
		runner.expect(r1.selected_armor_id == r2.selected_armor_id, "selected deterministic");
		runner.expect(d1.ballistic_arrival_time_s == d2.ballistic_arrival_time_s,
		              "ballistic_arrival deterministic");

		// 非收敛仍返回 Success（revision #4）。
		auto config1 = auto_aim::make_default_aimer_config();
		config1.max_refinement_iterations = 1;
		auto_aim::Aimer a3(config1);
		const auto r3 = a3.aim(moving, 1.0, 23.0);
		runner.expect(r3.valid && r3.status == auto_aim::AimStatus::Success,
		              "non-converged still Success");

		runner.end();
	}

	void test_predictive_advance(TestRunner& runner)
	{
		runner.begin("Predictive switch advance");

		auto config = make_predictive_config(0.3, 1.0); // hysteresis 0.3, max_advance 1.0

		auto run = [&](double yaw_rate) {
			const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
			                                0.0, yaw_rate, 0.2, 0.0, 0.0, 4,
			                                auto_aim::ArmorName::Four, 0.0, true, 1);

			auto_aim::Aimer aimer(config);
			auto_aim::AimerDebugData debug;
			const auto r = aimer.aim(target, 1.0, 23.0, &debug);
			runner.expect(r.valid, "valid");
			return debug.switch_advance_s;
		};

		runner.expect(near(run(3.0), 0.1), "advance(+3) == hysteresis/abs(w) == 0.1");
		runner.expect(near(run(-3.0), 0.1), "advance(-3) == 0.1 (symmetric)");
		runner.expect(near(run(0.0), 0.0), "advance(0) == 0 (no divide by zero)");
		runner.expect(near(run(0.1), 1.0), "advance clamped to max_advance");

		runner.end();
	}

	void test_predictive_aim_point_not_advanced(TestRunner& runner)
	{
		runner.begin("Predictive aim point not advanced");

		auto config = make_predictive_config(0.3, 1.0);
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 2.0, 0.2, 0.0, 0.0, 4,
		                                auto_aim::ArmorName::Four, 0.0, true, 1);

		auto_aim::AimerDebugData debug;
		const auto r = aimer.aim(target, 1.0, 23.0, &debug);

		runner.expect(r.valid, "valid");
		runner.expect(r.selected_armor_id.has_value(), "selected id present");

		const int sid = *r.selected_armor_id;

		// aim_point 应等于 armors_at_predict[sid]（t_predict 时刻）。
		const double dt_predict = debug.target_prediction_time_s - target.timestamp_s;
		const auto vehicle_predict = auto_aim::predict_vehicle(target, dt_predict);
		const auto armors_predict = auto_aim::armor_hypotheses(vehicle_predict);
		runner.expect(
		    vector_near(debug.aim_point_in_world, armors_predict[static_cast<std::size_t>(sid)].position_in_world,
		                1e-12),
		    "aim point == armor at target_prediction_time");

		// aim_point 不应等于 armors_at_selection[sid]（t_selection 时刻）。
		const double dt_selection = debug.armor_selection_time_s - target.timestamp_s;
		const auto vehicle_selection = auto_aim::predict_vehicle(target, dt_selection);
		const auto armors_selection = auto_aim::armor_hypotheses(vehicle_selection);
		runner.expect(
		    !vector_near(debug.aim_point_in_world,
		                 armors_selection[static_cast<std::size_t>(sid)].position_in_world, 1e-9),
		    "aim point != armor at selection time");

		runner.end();
	}

	void test_predictive_time_semantics(TestRunner& runner)
	{
		runner.begin("Predictive debug time semantics");

		auto config = make_predictive_config(0.3, 1.0);
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 2.0, 0.2, 0.0, 0.0, 4,
		                                auto_aim::ArmorName::Four, 0.0, true, 1);

		auto_aim::AimerDebugData debug;
		const auto r = aimer.aim(target, 1.0, 23.0, &debug);

		runner.expect(r.valid, "valid");
		runner.expect(near(debug.armor_selection_time_s,
		                   debug.target_prediction_time_s + debug.switch_advance_s, 1e-12),
		              "armor_selection == target_prediction + advance");
		runner.expect(near(debug.ballistic_arrival_time_s, debug.t_muzzle_s + debug.flight_time_s,
		                   1e-12),
		              "ballistic_arrival == t_muzzle + flight_time");

		runner.end();
	}

	void test_predictive_has_armor_switch_false(TestRunner& runner)
	{
		runner.begin("Predictive has_armor_switch false");

		auto config = make_predictive_config(0.3, 1.0);
		auto_aim::Aimer aimer(config);

		// has_armor_switch=false：无论哪块板 score 最低，恒选 id0。
		const auto target = make_target(Eigen::Vector3d(0.0, 3.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 0.0, 0.0, 4,
		                                auto_aim::ArmorName::Four, 0.0, false, 1);

		const auto r = aimer.aim(target, 1.0, 23.0);
		runner.expect(r.valid && r.selected_armor_id == 0, "has_armor_switch=false -> id 0");

		runner.end();
	}

	void test_predictive_hysteresis(TestRunner& runner)
	{
		runner.begin("Predictive hysteresis (no-switch / switch)");

		const double r = 3.0;
		const auto make_center = [&](double theta) {
			return Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), 0.0);
		};
		const auto make_t = [&](double theta) {
			return make_target(make_center(theta), Eigen::Vector3d::Zero(), 0.0, 0.0,
			                   0.2, 0.0, 0.0, 4, auto_aim::ArmorName::Four, 0.0, true, 1);
		};

		// no-switch：hysteresis=0.15，θ: π/4 -> π/4+0.05，best 变为 id1 但不够优 -> 保持 id0。
		{
			auto config = make_predictive_config(0.15, 1.0);
			auto_aim::Aimer aimer(config);
			const auto r1 = aimer.aim(make_t(kPi / 4.0), 1.0, 23.0);
			runner.expect(r1.selected_armor_id == 0, "frame1 (tie) -> id 0");
			const auto r2 = aimer.aim(make_t(kPi / 4.0 + 0.05), 1.0, 23.0);
			runner.expect(r2.selected_armor_id == 0, "no-switch: hysteresis keeps id 0");
		}

		// switch：hysteresis=0.05，best=id1 明显优于 id0 -> 切到 id1。
		{
			auto config = make_predictive_config(0.05, 1.0);
			auto_aim::Aimer aimer(config);
			const auto r1 = aimer.aim(make_t(kPi / 4.0), 1.0, 23.0);
			runner.expect(r1.selected_armor_id == 0, "frame1 (tie) -> id 0");
			const auto r2 = aimer.aim(make_t(kPi / 4.0 + 0.05), 1.0, 23.0);
			runner.expect(r2.selected_armor_id == 1, "switch: hysteresis allows id 1");
		}

		runner.end();
	}

	void test_predictive_early_switch(TestRunner& runner)
	{
		runner.begin("Predictive early switch");

		auto config = make_predictive_config(0.3, 1.0);
		auto_aim::Aimer aimer(config);

		// 旋转目标：center=(0,3,0) (center_yaw=π/2)，yaw0=0，yaw_rate=5。
		const auto target = make_target(Eigen::Vector3d(0.0, 3.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 5.0, 0.2, 0.0, 0.0, 4,
		                                auto_aim::ArmorName::Four, 0.0, true, 1);

		auto_aim::AimerDebugData debug;
		const auto r = aimer.aim(target, 0.0, 23.0, &debug);

		runner.expect(r.valid, "valid");
		runner.expect(debug.switch_advance_s > 0.0, "advance > 0");
		runner.expect(r.selected_armor_id.has_value(), "selected id present");

		const int sid = *r.selected_armor_id;

		const double dt_predict = debug.target_prediction_time_s - target.timestamp_s;
		const auto vehicle_predict = auto_aim::predict_vehicle(target, dt_predict);
		const double center_yaw = std::atan2(vehicle_predict.center.y(), vehicle_predict.center.x());

		auto best_at = [&](double t) {
			const auto vehicle = auto_aim::predict_vehicle(target, t - target.timestamp_s);
			const auto armors = auto_aim::armor_hypotheses(vehicle);
			int best = 0;
			double best_score = std::numeric_limits<double>::infinity();
			for(int i = 0; i < static_cast<int>(armors.size()); ++i)
			{
				const double score = std::abs(tools::maths_tools::limit_rad(
				    armors[static_cast<std::size_t>(i)].yaw_in_world - center_yaw));
				if(score < best_score)
				{
					best_score = score;
					best = i;
				}
			}
			return best;
		};

		const int best_predict = best_at(debug.target_prediction_time_s);
		const int best_selection = best_at(debug.armor_selection_time_s);

		runner.expect(sid == best_selection, "selected == best at t_selection");
		runner.expect(best_selection != best_predict,
		              "rotation flips best between t_predict and t_selection");

		runner.end();
	}

	void test_predictive_transaction(TestRunner& runner)
	{
		runner.begin("Predictive transaction (rollback)");

		auto config = make_predictive_config(0.15, 1.0);
		auto_aim::Aimer aimer(config);

		const double r = 3.0;
		const auto make_theta = [&](double theta) {
			return make_target(Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), 0.0),
			                   Eigen::Vector3d::Zero(), 0.0, 0.0, 0.2, 0.0, 0.0, 4,
			                   auto_aim::ArmorName::Four, 0.0, true, 1);
		};

		// frame 1：提交 id 0。
		auto_aim::AimerDebugData d1;
		const auto s1 = aimer.aim(make_theta(kPi / 4.0), 1.0, 23.0, &d1);
		runner.expect(s1.valid && s1.selected_armor_id == 0, "frame1 commit id 0");

		// frame 2：过远 -> 选板成功但弹道不可解（中间会选 id1），失败不提交。
		const auto far_target = make_target(Eigen::Vector3d(0.0, 100.0, 0.0),
		                                    Eigen::Vector3d::Zero(), 0.0, 0.0, 0.2, 0.0, 0.0, 4,
		                                    auto_aim::ArmorName::Four, 0.0, true, 1);
		auto_aim::AimerDebugData d2;
		const auto s2 = aimer.aim(far_target, 1.0, 23.0, &d2);
		runner.expect(!s2.valid && s2.status == auto_aim::AimStatus::BallisticUnsolvable,
		              "frame2 ballistic unsolvable");

		// frame 3：成功，previous 应仍为 0（frame2 未提交）。
		auto_aim::AimerDebugData d3;
		const auto s3 = aimer.aim(make_theta(kPi / 4.0), 1.0, 23.0, &d3);
		runner.expect(s3.valid, "frame3 valid");
		runner.expect(d3.previous_predictive_armor_id == 0,
		              "frame2 failure did not commit (previous still 0)");

		runner.end();
	}

	void test_predictive_token_reset(TestRunner& runner)
	{
		runner.begin("Predictive target token reset");

		auto config = make_predictive_config(0.15, 1.0);
		auto_aim::Aimer aimer(config);

		const double r = 3.0;
		const auto make_t = [&](double theta, std::uint64_t token) {
			return make_target(Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), 0.0),
			                   Eigen::Vector3d::Zero(), 0.0, 0.0, 0.2, 0.0, 0.0, 4,
			                   auto_aim::ArmorName::Four, 0.0, true, token);
		};

		const auto s1 = aimer.aim(make_t(kPi / 4.0, 1), 1.0, 23.0);
		runner.expect(s1.selected_armor_id == 0, "token1 frame1 -> id0");

		const auto s2 = aimer.aim(make_t(kPi / 4.0 + 0.05, 1), 1.0, 23.0);
		runner.expect(s2.selected_armor_id == 0, "token1 frame2 keeps id0 (hysteresis)");

		const auto s3 = aimer.aim(make_t(kPi / 4.0 + 0.05, 2), 1.0, 23.0);
		runner.expect(s3.selected_armor_id == 1, "new token resets predictive -> best id1");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_aimer");
	std::printf("=== Aimer Test Suite ===\n\n");

	TestRunner runner;

	test_bullet_speed_fallback(runner);
	test_bullet_speed_fail_safe(runner);
	test_delay_signed(runner);
	test_gyro_compatibility(runner);
	test_has_armor_switch(runner);
	test_lock_sequence(runner);
	test_fixed_point_deterministic(runner);
	test_predictive_advance(runner);
	test_predictive_aim_point_not_advanced(runner);
	test_predictive_time_semantics(runner);
	test_predictive_has_armor_switch_false(runner);
	test_predictive_hysteresis(runner);
	test_predictive_early_switch(runner);
	test_predictive_transaction(runner);
	test_predictive_token_reset(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Aimer tests failed ===\n");
		return 1;
	}

	std::printf("=== All aimer tests passed ===\n");
	return 0;
}
