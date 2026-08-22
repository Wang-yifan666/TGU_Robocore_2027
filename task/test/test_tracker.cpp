/**
 * @file test_tracker.cpp
 * @brief Tracker 状态机确定性单元测试（纯 synthetic ArmorObservation）。
 */

#include "app/auto_aim/tracker.hpp"

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

	using namespace app::auto_aim;

	TrackerConfig default_config()
	{
		TrackerConfig c;
		c.detecting_confirm_hits = 3;
		c.detecting_max_misses = 1;
		c.temp_lost_max_misses = 2;
		c.max_dt_s = 0.5;

		c.association.max_position_error_m = 0.5;
		c.association.max_yaw_error_rad = 0.5;
		c.association.position_score_scale_m = 1.0;
		c.association.yaw_score_scale_rad = 1.0;

		c.initial_covariance = Eigen::MatrixXd::Identity(kTargetStateDim, kTargetStateDim);
		c.measurement_covariance =
		    1e-4 * Eigen::MatrixXd::Identity(kTargetMeasurementDim, kTargetMeasurementDim);

		c.process_noise.translation_accel_variance = 1.0;
		c.process_noise.yaw_accel_variance = 1.0;
		c.process_noise.radius_random_walk_variance = 1.0;
		c.process_noise.delta_radius_random_walk_variance = 1.0;
		c.process_noise.delta_z_random_walk_variance = 1.0;

		c.min_radius_m = 0.05;
		c.max_radius_m = 0.5;

		return c;
	}

	// 静止 4-armor 车辆：center (0,0,0), yaw=0, radius=0.2。
	// armor 0 位置 (-0.2, 0, 0), yaw 0。
	ArmorObservation obs_armor0(double timestamp = 0.0, ArmorPriority priority = ArmorPriority::First)
	{
		ArmorObservation o;
		o.color = ArmorColor::Red;
		o.name = ArmorName::Four;
		o.type = ArmorType::Small;
		o.priority = priority;
		o.position_in_world = Eigen::Vector3d(-0.2, 0.0, 0.0);
		o.armor_yaw_in_world = 0.0;
		o.timestamp_s = timestamp;
		o.source_detection_index = 0;
		return o;
	}

	// 静止 4-armor 车辆 armor 2 位置 (0.2, 0, 0), yaw = pi（用 -pi 精确表示）。
	ArmorObservation obs_armor2(double timestamp = 0.0, ArmorPriority priority = ArmorPriority::First)
	{
		ArmorObservation o;
		o.color = ArmorColor::Red;
		o.name = ArmorName::Four;
		o.type = ArmorType::Small;
		o.priority = priority;
		o.position_in_world = Eigen::Vector3d(0.2, 0.0, 0.0);
		o.armor_yaw_in_world = -kPi;
		o.timestamp_s = timestamp;
		o.source_detection_index = 0;
		return o;
	}

	// ============================================================
	// Test：Lost empty
	// ============================================================

	void test_lost_empty(TestRunner& runner)
	{
		runner.begin("Lost empty");

		Tracker tracker(default_config());

		runner.expect(!tracker.track({}, 0.0).has_value(), "empty -> nullopt (Lost)");

		runner.end();
	}

	// ============================================================
	// Test：Lost -> Detecting -> Tracking (exact frame count)
	// ============================================================

	void test_detecting_to_tracking(TestRunner& runner)
	{
		runner.begin("Detecting to Tracking (exact frame count)");

		Tracker tracker(default_config());

		// frame 1：Lost -> Detecting（hit=1）。
		auto r1 = tracker.track({obs_armor0(0.0)}, 0.0);
		runner.expect(r1 && r1->state == TrackerState::Detecting, "frame1 Detecting");

		// frame 2：hit=2，仍 Detecting。
		auto r2 = tracker.track({obs_armor0(0.1)}, 0.1);
		runner.expect(r2 && r2->state == TrackerState::Detecting, "frame2 Detecting");

		// frame 3：hit=3 -> Tracking。
		auto r3 = tracker.track({obs_armor0(0.2)}, 0.2);
		runner.expect(r3 && r3->state == TrackerState::Tracking, "frame3 Tracking");

		runner.end();
	}

	// ============================================================
	// Test：Tracking remains Tracking while hits continue
	// ============================================================

	void test_tracking_stays(TestRunner& runner)
	{
		runner.begin("Tracking stays Tracking");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		auto r = tracker.track({obs_armor0(0.3)}, 0.3);
		runner.expect(r && r->state == TrackerState::Tracking, "still Tracking");
		runner.expect(r && r->has_measurement, "has_measurement true");

		runner.end();
	}

	// ============================================================
	// Test：Tracking miss -> TempLost
	// ============================================================

	void test_tracking_miss_temp_lost(TestRunner& runner)
	{
		runner.begin("Tracking miss -> TempLost");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		// miss：空观测（association 失败）。
		auto r = tracker.track({}, 0.3);
		runner.expect(r && r->state == TrackerState::TempLost, "miss -> TempLost");
		runner.expect(r && !r->has_measurement, "has_measurement false in TempLost");

		runner.end();
	}

	// ============================================================
	// Test：TempLost predicts with has_measurement=false, reacquire, timeout
	// ============================================================

	void test_temp_lost_predict_reacquire_lost(TestRunner& runner)
	{
		runner.begin("TempLost predict / reacquire / lost");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		// 进入 TempLost。
		auto r1 = tracker.track({}, 0.3);
		runner.expect(r1 && r1->state == TrackerState::TempLost, "TempLost after miss");
		runner.expect(r1 && !r1->has_measurement, "TempLost has_measurement false");

		// reacquire。
		auto r2 = tracker.track({obs_armor0(0.4)}, 0.4);
		runner.expect(r2 && r2->state == TrackerState::Tracking, "reacquire -> Tracking");
		runner.expect(r2 && r2->has_measurement, "reacquired has_measurement true");

		runner.end();
	}

	void test_temp_lost_timeout(TestRunner& runner)
	{
		runner.begin("TempLost timeout -> Lost");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		// 进入 TempLost（miss#1）。
		tracker.track({}, 0.3);
		// miss#2。
		tracker.track({}, 0.4);
		// miss#3 超过 temp_lost_max_misses=2 -> Lost。
		auto r = tracker.track({}, 0.5);
		runner.expect(!r.has_value(), "TempLost timeout -> Lost (nullopt)");

		runner.end();
	}

	// ============================================================
	// Test：Detecting miss policy
	// ============================================================

	void test_detecting_miss_policy(TestRunner& runner)
	{
		runner.begin("Detecting miss policy");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0); // Detecting hit=1

		// miss#1：detecting_max_misses=1，超过则 Lost。此处 miss 后 miss_count=1，
		// 由于判定是 >，第二次 miss 才会退回 Lost。
		auto r1 = tracker.track({}, 0.1);
		runner.expect(r1 && r1->state == TrackerState::Detecting, "first detecting miss still Detecting");

		// miss#2：> detecting_max_misses -> Lost。
		auto r2 = tracker.track({}, 0.2);
		runner.expect(!r2.has_value(), "second detecting miss -> Lost");

		runner.end();
	}

	// ============================================================
	// Test：reset -> Lost
	// ============================================================

	void test_reset(TestRunner& runner)
	{
		runner.begin("Reset");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		tracker.reset();
		runner.expect(!tracker.track({}, 0.3).has_value(), "after reset empty -> nullopt");

		runner.end();
	}

	// ============================================================
	// Test：dt == 0
	// ============================================================

	void test_dt_zero(TestRunner& runner)
	{
		runner.begin("dt == 0");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		// 同 timestamp，dt == 0，允许 association/correction，不除以 dt。
		auto r = tracker.track({obs_armor0(0.0)}, 0.0);
		runner.expect(r.has_value(), "dt==0 still processes");
		runner.expect(r->state == TrackerState::Detecting || r->state == TrackerState::Tracking,
		              "dt==0 state valid");

		runner.end();
	}

	// ============================================================
	// Test：dt < 0 -> reset + reinit
	// ============================================================

	void test_dt_negative(TestRunner& runner)
	{
		runner.begin("dt < 0 reset + reinit");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);

		// 时间回退：reset 后当前帧重新初始化。
		auto r = tracker.track({obs_armor0(-0.1)}, -0.1);
		runner.expect(r && r->state == TrackerState::Detecting, "dt<0 reinit -> Detecting");

		runner.end();
	}

	// ============================================================
	// Test：dt > max_dt -> reset + reinit
	// ============================================================

	void test_dt_excessive(TestRunner& runner)
	{
		runner.begin("dt > max_dt reset + reinit");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);

		// 远超 max_dt_s=0.5。
		auto r = tracker.track({obs_armor0(2.0)}, 2.0);
		runner.expect(r && r->state == TrackerState::Detecting, "dt>max reset -> Detecting");

		runner.end();
	}

	// ============================================================
	// Test：non-finite timestamp
	// ============================================================

	void test_nonfinite_timestamp(TestRunner& runner)
	{
		runner.begin("Non-finite timestamp");

		Tracker tracker(default_config());

		bool threw = false;

		try
		{
			tracker.track({obs_armor0()}, std::numeric_limits<double>::quiet_NaN());
		}
		catch(const std::invalid_argument&)
		{
			threw = true;
		}

		runner.expect(threw, "non-finite timestamp throws invalid_argument");

		runner.end();
	}

	// ============================================================
	// Test：initialization priority selection
	// ============================================================

	void test_init_priority(TestRunner& runner)
	{
		runner.begin("Initialization priority");

		Tracker tracker(default_config());

		// 两个 observation：One（First）与 Three（Second）。应选 First(One)。
		auto obs_first = obs_armor0(0.0, ArmorPriority::First);
		auto obs_second = obs_armor0(0.0, ArmorPriority::Second);
		obs_second.name = ArmorName::Three;

		const auto r = tracker.track({obs_second, obs_first}, 0.0);
		runner.expect(r && r->state == TrackerState::Detecting, "init -> Detecting");
		runner.expect(r && r->name == ArmorName::Four, "selected First priority observation (Four)");

		runner.end();
	}

	// ============================================================
	// Test：stale NIS not exposed
	// ============================================================

	void test_stale_nis_not_exposed(TestRunner& runner)
	{
		runner.begin("Stale NIS not exposed");

		Tracker tracker(default_config());

		// frame1：Detecting 无 measurement。
		auto r1 = tracker.track({obs_armor0(0.0)}, 0.0);
		runner.expect(r1 && !r1->has_measurement, "frame1 no measurement");
		runner.expect(r1 && !r1->nis.has_value(), "frame1 NIS nullopt");

		// 后续帧有 measurement。
		auto r2 = tracker.track({obs_armor0(0.1)}, 0.1);
		runner.expect(r2 && r2->has_measurement, "frame2 measurement");

		// 在一次 miss 后，has_measurement 应变 false，且不暴露 stale NIS。
		// 先确认进入 Tracking。
		tracker.track({obs_armor0(0.2)}, 0.2);
		auto r_miss = tracker.track({}, 0.3);
		runner.expect(r_miss && !r_miss->has_measurement, "miss frame no measurement");
		runner.expect(r_miss && !r_miss->nis.has_value(), "miss frame NIS nullopt (no stale)");

		runner.end();
	}

	// ============================================================
	// Test：board switching continuity
	// ============================================================

	void test_board_switch_continuity(TestRunner& runner)
	{
		runner.begin("Board switch continuity");

		Tracker tracker(default_config());

		tracker.track({obs_armor0(0.0)}, 0.0);
		tracker.track({obs_armor0(0.1)}, 0.1);
		tracker.track({obs_armor0(0.2)}, 0.2);

		const Eigen::Vector3d center_before = [&]() {
			auto r = tracker.track({obs_armor0(0.3)}, 0.3);
			return r->center_in_world;
		}();

		// 切换到对侧 armor 2：center 不应跳。
		auto r_switch = tracker.track({obs_armor2(0.4)}, 0.4);
		runner.expect(r_switch.has_value(), "switch accepted");

		const double jump = (r_switch->center_in_world - center_before).norm();
		runner.expect(jump < 0.05, "vehicle center does not jump across armor switch");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Tracker State Machine Test Suite ===\n\n");

	TestRunner runner;

	test_lost_empty(runner);
	test_detecting_to_tracking(runner);
	test_tracking_stays(runner);
	test_tracking_miss_temp_lost(runner);
	test_temp_lost_predict_reacquire_lost(runner);
	test_temp_lost_timeout(runner);
	test_detecting_miss_policy(runner);
	test_reset(runner);
	test_dt_zero(runner);
	test_dt_negative(runner);
	test_dt_excessive(runner);
	test_nonfinite_timestamp(runner);
	test_init_priority(runner);
	test_stale_nis_not_exposed(runner);
	test_board_switch_continuity(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Tracker tests failed ===\n");
		return 1;
	}

	std::printf("=== All tracker tests passed ===\n");
	return 0;
}