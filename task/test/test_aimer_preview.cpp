/**
 * @file test_aimer_preview.cpp
 * @brief Aimer preview（PlannerPreviewSeed + sample_at）单元测试。
 */

#include "app/auto_aim/aimer.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

#include "test_logging.hpp"

namespace
{

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

	namespace auto_aim = app::auto_aim;

	auto_aim::TrackedTarget make_target(Eigen::Vector3d center, Eigen::Vector3d velocity,
	                                    double yaw, double yaw_rate, double radius, int armor_count,
	                                    std::uint64_t token = 1, bool has_armor_switch = false)
	{
		auto_aim::TrackedTarget t;
		t.state = auto_aim::TrackerState::Tracking;
		t.timestamp_s = 0.0;
		t.name = auto_aim::ArmorName::Four;
		t.center_in_world = center;
		t.velocity_in_world = velocity;
		t.yaw = yaw;
		t.yaw_rate = yaw_rate;
		t.radius = radius;
		t.delta_radius = 0.0;
		t.delta_z = 0.0;
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

	void test_seed_time_and_bullet(TestRunner& runner)
	{
		runner.begin("seed time / effective bullet speed");

		const auto config = auto_aim::make_default_aimer_config(); // fallback=23, delay=0
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		auto_aim::PlannerPreviewSeed seed;
		const double nan = std::numeric_limits<double>::quiet_NaN();
		const auto aiming = aimer.aim(target, 1.0, nan, nullptr, &seed);

		runner.expect(aiming.valid, "aim valid (fallback)");
		runner.expect(near(seed.effective_bullet_speed_mps, 23.0, 1e-12),
		              "effective bullet speed == fallback 23.0");
		runner.expect(seed.reference_center_time_s > 1.0, "reference_center_time > t_now");
		runner.expect(seed.target_token == 1, "seed token == target token");

		runner.end();
	}

	void test_center_equivalence(TestRunner& runner)
	{
		runner.begin("center sample == raw AimingSolution");

		const auto config = auto_aim::make_default_aimer_config();
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		auto_aim::PlannerPreviewSeed seed;
		const auto aiming = aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		runner.expect(aiming.valid, "aim valid");

		auto_aim::AimerPreviewState state = seed.preview_state;
		const auto sample = aimer.sample_at(state, target, seed.reference_center_time_s,
		                                    seed.effective_bullet_speed_mps);

		runner.expect(sample.valid, "center sample valid");
		runner.expect(near(sample.yaw_rad, seed.reference_center_yaw_rad, 1e-9),
		              "center sample yaw == raw aiming yaw");
		runner.expect(near(sample.pitch_rad, seed.reference_center_pitch_rad, 1e-9),
		              "center sample pitch == raw aiming pitch");

		runner.end();
	}

	void test_state_isolation(TestRunner& runner)
	{
		runner.begin("preview does not pollute Aimer state");

		const auto config = auto_aim::make_default_aimer_config();
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4);

		const auto before = aimer.aim(target, 1.0, 23.0);

		auto_aim::PlannerPreviewSeed seed;
		aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		auto_aim::AimerPreviewState state = seed.preview_state;

		for(int i = 0; i < 200; ++i)
		{
			const double t = seed.reference_center_time_s + (i - 100) * 0.01;
			aimer.sample_at(state, target, t, seed.effective_bullet_speed_mps);
		}

		const auto after = aimer.aim(target, 1.0, 23.0);
		runner.expect(near(before.yaw_rad, after.yaw_rad, 1e-9), "yaw unchanged after preview");
		runner.expect(near(before.pitch_rad, after.pitch_rad, 1e-9), "pitch unchanged after preview");

		runner.end();
	}

	void test_hysteresis_evolution(TestRunner& runner)
	{
		runner.begin("local hysteresis evolution");

		auto config = auto_aim::make_default_aimer_config();
		config.armor_switch_strategy = auto_aim::ArmorSwitchStrategy::PredictiveHysteresis;
		config.predictive_switch_hysteresis_rad = 0.3;
		config.predictive_switch_max_advance_s = 0.2;
		auto_aim::Aimer aimer(config);

		const auto target = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                0.0, 0.0, 0.2, 4, 1, true);

		auto_aim::PlannerPreviewSeed seed;
		const auto aiming = aimer.aim(target, 1.0, 23.0, nullptr, &seed);
		runner.expect(aiming.valid, "aim valid");

		auto_aim::AimerPreviewState state = seed.preview_state;
		auto sample = aimer.sample_at(state, target, seed.reference_center_time_s,
		                              seed.effective_bullet_speed_mps);
		runner.expect(sample.valid, "sample valid");
		runner.expect(sample.selected_armor_id.has_value(), "sample has selected id");
		runner.expect(state.predictive_selected_armor_id.has_value(),
		              "local predictive id evolved after sample");

		runner.end();
	}

	void test_token_scope(TestRunner& runner)
	{
		runner.begin("target token scope");

		const auto config = auto_aim::make_default_aimer_config();
		auto_aim::Aimer aimer(config);

		const auto target1 = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                 0.0, 0.0, 0.2, 4, 1, true);
		auto_aim::PlannerPreviewSeed seed1;
		aimer.aim(target1, 1.0, 23.0, nullptr, &seed1);

		const auto target2 = make_target(Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d::Zero(),
		                                 0.0, 0.0, 0.2, 4, 2, true);
		auto_aim::PlannerPreviewSeed seed2;
		aimer.aim(target2, 1.0, 23.0, nullptr, &seed2);

		runner.expect(seed2.target_token == 2, "seed2 token == 2");
		runner.expect(seed1.target_token != seed2.target_token, "seed token scoped per target");

		runner.end();
	}
} // namespace

int main()
{
	test_logging::init("test_aimer_preview");
	std::printf("=== Aimer Preview Test Suite ===\n\n");

	TestRunner runner;
	test_seed_time_and_bullet(runner);
	test_center_equivalence(runner);
	test_state_isolation(runner);
	test_hysteresis_evolution(runner);
	test_token_scope(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Aimer preview tests failed ===\n");
		return 1;
	}

	std::printf("=== All Aimer preview tests passed ===\n");
	return 0;
}
