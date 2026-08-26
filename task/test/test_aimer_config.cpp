/**
 * @file test_aimer_config.cpp
 * @brief AimerConfig TOML loader 单元测试（内存 toml::parse + load_aimer_config_from_table）。
 *
 * 覆盖：
 * - 合法 table -> success；
 * - 缺失 [aimer] 表 / 缺失字段 -> false；
 * - 非法 enum 字符串 -> false；
 * - NaN / Inf -> false；
 * - 负值 -> false；
 * - 弹速 <= 0 -> false；
 * - max_refinement_iterations < 1 -> false；
 * - make_default_aimer_config() 兼容默认值。
 */

#include "app/auto_aim/aimer_config.hpp"
#include "tools/tomlpp.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>

namespace
{

	namespace auto_aim = app::auto_aim;

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
	// TOML 片段
	// ============================================================

	const char* kValidToml = R"(
[aimer]
yaw_offset_rad = 0.0
pitch_offset_rad = 0.0
coming_angle_rad = 1.0471204188481676
leaving_angle_rad = 0.3490401396160559
outpost_coming_angle_rad = 1.2216404886569256
outpost_leaving_angle_rad = 0.5235602094240838
shootable_angle_threshold_rad = 1.0471204188481676
high_speed_delay_s = 0.0
low_speed_delay_s = 0.0
decision_speed_rad_s = 10.0
use_radius_for_gyro_detection = true
non_gyro_radius_threshold_m = 2.0
non_gyro_yaw_rate_threshold_rad_s = 2.0
invalid_bullet_speed_policy = "fallback"
min_valid_bullet_speed_mps = 14.0
fallback_bullet_speed_mps = 23.0
max_refinement_iterations = 10
flight_time_convergence_s = 0.001
armor_switch_strategy = "sp_compat"
)";

	std::string replace(std::string source, std::string_view from, std::string_view to)
	{
		const auto pos = source.find(from);

		if(pos != std::string::npos)
		{
			source.replace(pos, from.size(), to);
		}

		return source;
	}

	bool load_from_string(std::string_view toml_text, auto_aim::AimerConfig& config)
	{
		const toml::table root = toml::parse(toml_text);
		return auto_aim::load_aimer_config_from_table(root, config);
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_valid_config(TestRunner& runner)
	{
		runner.begin("Valid config");

		auto_aim::AimerConfig config;

		runner.expect(load_from_string(kValidToml, config), "valid TOML should load");
		runner.expect(near(config.decision_speed_rad_s, 10.0), "decision_speed_rad_s");
		runner.expect(config.use_radius_for_gyro_detection, "use_radius_for_gyro_detection == true");
		runner.expect(config.invalid_bullet_speed_policy
		                  == auto_aim::InvalidBulletSpeedPolicy::Fallback,
		              "invalid_bullet_speed_policy == fallback");
		runner.expect(config.max_refinement_iterations == 10, "max_refinement_iterations == 10");
		runner.expect(config.armor_switch_strategy == auto_aim::ArmorSwitchStrategy::SpCompat,
		              "armor_switch_strategy == sp_compat");

		runner.end();
	}

	void test_missing_table_and_field(TestRunner& runner)
	{
		runner.begin("Missing table / field");

		auto_aim::AimerConfig config;

		runner.expect(!load_from_string("[aimer]\n", config), "empty [aimer] should fail");

		const std::string missing_yaw = replace(kValidToml, "yaw_offset_rad = 0.0\n", "");
		runner.expect(!load_from_string(missing_yaw, config), "missing yaw_offset_rad should fail");

		runner.end();
	}

	void test_invalid_enum(TestRunner& runner)
	{
		runner.begin("Invalid enum");

		auto_aim::AimerConfig config;

		runner.expect(!load_from_string(replace(kValidToml, "\"fallback\"", "\"bogus\""), config),
		              "invalid bullet speed policy should fail");
		runner.expect(!load_from_string(replace(kValidToml, "\"sp_compat\"", "\"bogus\""), config),
		              "invalid armor switch strategy should fail");

		runner.end();
	}

	void test_non_finite_values(TestRunner& runner)
	{
		runner.begin("Non-finite values");

		auto_aim::AimerConfig config;

		runner.expect(!load_from_string(replace(kValidToml, "decision_speed_rad_s = 10.0",
		                                        "decision_speed_rad_s = nan"),
		                                config),
		              "NaN decision_speed_rad_s should fail");
		runner.expect(!load_from_string(replace(kValidToml, "fallback_bullet_speed_mps = 23.0",
		                                        "fallback_bullet_speed_mps = inf"),
		                                config),
		              "Inf fallback_bullet_speed_mps should fail");

		runner.end();
	}

	void test_negative_values(TestRunner& runner)
	{
		runner.begin("Negative values");

		auto_aim::AimerConfig config;

		runner.expect(!load_from_string(replace(kValidToml, "high_speed_delay_s = 0.0",
		                                        "high_speed_delay_s = -0.1"),
		                                config),
		              "negative high_speed_delay_s should fail");
		runner.expect(!load_from_string(replace(kValidToml, "min_valid_bullet_speed_mps = 14.0",
		                                        "min_valid_bullet_speed_mps = -1.0"),
		                                config),
		              "negative min_valid_bullet_speed_mps should fail");
		runner.expect(!load_from_string(replace(kValidToml, "max_refinement_iterations = 10",
		                                        "max_refinement_iterations = 0"),
		                                config),
		              "max_refinement_iterations == 0 should fail");

		runner.end();
	}

	void test_compatibility_defaults(TestRunner& runner)
	{
		runner.begin("Compatibility defaults");

		const auto_aim::AimerConfig config = auto_aim::make_default_aimer_config();

		runner.expect(config.use_radius_for_gyro_detection, "use_radius_for_gyro_detection == true");
		runner.expect(near(config.non_gyro_radius_threshold_m, 2.0), "radius threshold == 2.0 m");
		runner.expect(near(config.min_valid_bullet_speed_mps, 14.0), "min bullet speed == 14 m/s");
		runner.expect(near(config.fallback_bullet_speed_mps, 23.0), "fallback bullet speed == 23 m/s");
		runner.expect(config.max_refinement_iterations == 10, "max_refinement_iterations == 10");
		runner.expect(near(config.flight_time_convergence_s, 0.001), "convergence == 0.001 s");
		runner.expect(near(config.shootable_angle_threshold_rad, 60.0 / 57.3),
		              "shootable angle == 60 / 57.3 rad");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== AimerConfig Loader Test Suite ===\n\n");

	TestRunner runner;

	test_valid_config(runner);
	test_missing_table_and_field(runner);
	test_invalid_enum(runner);
	test_non_finite_values(runner);
	test_negative_values(runner);
	test_compatibility_defaults(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== AimerConfig loader tests failed ===\n");
		return 1;
	}

	std::printf("=== All aimer_config tests passed ===\n");
	return 0;
}
