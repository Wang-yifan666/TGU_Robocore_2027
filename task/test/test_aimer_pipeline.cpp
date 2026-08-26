/**
 * @file test_aimer_pipeline.cpp
 * @brief Tracker -> Aimer 链路集成测试（synthetic ArmorObservation）。
 *
 * 验证 TrackedTarget 快照能直接驱动 Aimer，并观察 selected_armor_id /
 * yaw / pitch / fire_allowed / has_armor_switch 传播。
 */

#include "app/auto_aim/aimer.hpp"
#include "app/auto_aim/tracker.hpp"
#include "tools/maths_tools.hpp"

#include <cmath>
#include <cstdio>
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

	// ============================================================
	// helper
	// ============================================================

	auto_aim::TrackerConfig tracker_config()
	{
		auto_aim::TrackerConfig c = auto_aim::make_default_tracker_config();
		c.detecting_confirm_hits = 1; // init 直接进入 Tracking。
		return c;
	}

	auto_aim::ArmorObservation obs_armor0(double timestamp_s)
	{
		auto_aim::ArmorObservation o;
		o.color = auto_aim::ArmorColor::Red;
		o.name = auto_aim::ArmorName::Four;
		o.type = auto_aim::ArmorType::Small;
		o.priority = auto_aim::ArmorPriority::First;
		o.position_in_world = Eigen::Vector3d(-0.2, 0.0, 0.0);
		o.armor_yaw_in_world = 0.0;
		o.ypd_in_world = tools::maths_tools::xyz2ypd(o.position_in_world);
		o.timestamp_s = timestamp_s;
		o.source_detection_index = 0;
		return o;
	}

	auto_aim::ArmorObservation obs_armor2(double timestamp_s)
	{
		auto_aim::ArmorObservation o;
		o.color = auto_aim::ArmorColor::Red;
		o.name = auto_aim::ArmorName::Four;
		o.type = auto_aim::ArmorType::Small;
		o.priority = auto_aim::ArmorPriority::First;
		o.position_in_world = Eigen::Vector3d(0.2, 0.0, 0.0);
		o.armor_yaw_in_world = -kPi;
		o.ypd_in_world = tools::maths_tools::xyz2ypd(o.position_in_world);
		o.timestamp_s = timestamp_s;
		o.source_detection_index = 0;
		return o;
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_pipeline(TestRunner& runner)
	{
		runner.begin("Tracker -> Aimer pipeline");

		auto_aim::Tracker tracker(tracker_config());
		auto_aim::Aimer aimer(auto_aim::make_default_aimer_config());

		// frame 1：init（armor 0）-> Tracking，has_armor_switch == false。
		const auto tr1 = tracker.track({obs_armor0(0.0)}, 0.0);
		runner.expect(tr1.target.has_value(), "tracker init produces target");

		const auto sol1 = aimer.aim(*tr1.target, 0.0, 23.0);
		runner.expect(sol1.valid, "aimer valid after init");
		runner.expect(sol1.selected_armor_id == 0, "no switch -> selected id 0");
		runner.expect(!sol1.fire_allowed, "fire_allowed false");
		runner.expect(std::isfinite(sol1.yaw_rad) && std::isfinite(sol1.pitch_rad),
		              "yaw/pitch finite");

		// frame 2：切到 armor 2 -> has_armor_switch == true。
		const auto tr2 = tracker.track({obs_armor2(0.1)}, 0.1);
		runner.expect(tr2.target.has_value(), "tracker keeps target after switch");
		runner.expect(tr2.target->has_armor_switch, "switch sets has_armor_switch");

		const auto sol2 = aimer.aim(*tr2.target, 0.1, 23.0);
		runner.expect(sol2.valid, "aimer valid after switch");
		runner.expect(sol2.selected_armor_id.has_value(), "selected armor present after switch");
		runner.expect(!sol2.fire_allowed, "fire_allowed false after switch");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== Aimer Pipeline Test Suite ===\n\n");

	TestRunner runner;

	test_pipeline(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Aimer pipeline tests failed ===\n");
		return 1;
	}

	std::printf("=== All aimer pipeline tests passed ===\n");
	return 0;
}
