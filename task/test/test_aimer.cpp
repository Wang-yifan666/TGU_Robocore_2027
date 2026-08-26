/**
 * @file test_aimer.cpp
 * @brief Aimer 单元测试（纯 synthetic TrackedTarget）。
 *
 * 覆盖：bullet speed、signed delay、gyro 兼容双模式、has_armor_switch、
 * armor lock 生命周期、fixed-point 收敛、deterministic 时间。
 */

#include "app/auto_aim/aimer.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

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

		runner.expect(!r23.fire_allowed, "fire_allowed stays false");

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
		runner.expect(!r.fire_allowed, "fire_allowed false");

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
		runner.expect(d1.t_hit_s > d1.t_muzzle_s, "t_hit > t_muzzle (positive flight time)");
		runner.expect(d1.refinement_iterations <= config.max_refinement_iterations,
		              "iterations <= max");

		// deterministic：两个全新 Aimer，相同输入 -> 逐位一致。
		runner.expect(r1.yaw_rad == r2.yaw_rad, "yaw deterministic");
		runner.expect(r1.pitch_rad == r2.pitch_rad, "pitch deterministic");
		runner.expect(r1.selected_armor_id == r2.selected_armor_id, "selected deterministic");
		runner.expect(d1.t_hit_s == d2.t_hit_s, "t_hit deterministic");

		// 非收敛仍返回 Success（revision #4）。
		auto config1 = auto_aim::make_default_aimer_config();
		config1.max_refinement_iterations = 1;
		auto_aim::Aimer a3(config1);
		const auto r3 = a3.aim(moving, 1.0, 23.0);
		runner.expect(r3.valid && r3.status == auto_aim::AimStatus::Success,
		              "non-converged still Success");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Aimer Test Suite ===\n\n");

	TestRunner runner;

	test_bullet_speed_fallback(runner);
	test_bullet_speed_fail_safe(runner);
	test_delay_signed(runner);
	test_gyro_compatibility(runner);
	test_has_armor_switch(runner);
	test_lock_sequence(runner);
	test_fixed_point_deterministic(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Aimer tests failed ===\n");
		return 1;
	}

	std::printf("=== All aimer tests passed ===\n");
	return 0;
}
