/**
 * @file test_shooter_config.cpp
 * @brief ShooterConfig TOML loader 单元测试（内存 toml::parse + load_shooter_config_from_table）。
 */

#include "app/auto_aim/shooter_config.hpp"
#include "tools/tomlpp.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>

#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

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

	bool near(double lhs, double rhs, double eps = 1e-12)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	const char* kValidToml = R"(
[shooter]
auto_fire = true
near_tolerance_rad = 0.08726003490401396
far_tolerance_rad = 0.034904013961605584
distance_threshold_m = 3.0
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

	bool load_from_string(std::string_view toml_text, auto_aim::ShooterConfig& config)
	{
		const toml::table root = toml::parse(toml_text);
		return auto_aim::load_shooter_config_from_table(root, config);
	}

	void test_valid_config(TestRunner& runner)
	{
		runner.begin("Valid config");

		auto_aim::ShooterConfig config;

		runner.expect(load_from_string(kValidToml, config), "valid toml loads");
		runner.expect(config.auto_fire, "auto_fire == true");
		runner.expect(near(config.near_tolerance_rad, 0.08726003490401396), "near tolerance");
		runner.expect(near(config.far_tolerance_rad, 0.034904013961605584), "far tolerance");
		runner.expect(near(config.distance_threshold_m, 3.0), "distance threshold");

		runner.end();
	}

	void test_missing_table_and_field(TestRunner& runner)
	{
		runner.begin("Missing table / field");

		auto_aim::ShooterConfig config;

		runner.expect(!load_from_string(replace(kValidToml, "[shooter]", "[other]"), config),
		              "missing [shooter] -> false");
		runner.expect(!load_from_string(replace(kValidToml, "auto_fire = true\n", ""), config),
		              "missing auto_fire -> false");
		runner.expect(
		    !load_from_string(replace(kValidToml, "near_tolerance_rad = 0.08726003490401396\n", ""),
		                      config),
		    "missing near_tolerance_rad -> false");
		runner.expect(
		    !load_from_string(replace(kValidToml, "far_tolerance_rad = 0.034904013961605584\n", ""),
		                      config),
		    "missing far_tolerance_rad -> false");
		runner.expect(
		    !load_from_string(replace(kValidToml, "distance_threshold_m = 3.0\n", ""), config),
		    "missing distance_threshold_m -> false");

		runner.end();
	}

	void test_invalid_auto_fire_type(TestRunner& runner)
	{
		runner.begin("Invalid auto_fire type");

		auto_aim::ShooterConfig config;

		runner.expect(!load_from_string(
		                  replace(kValidToml, "auto_fire = true", "auto_fire = \"yes\""), config),
		              "auto_fire string -> false");

		runner.end();
	}

	void test_non_finite_values(TestRunner& runner)
	{
		runner.begin("Non-finite values");

		auto_aim::ShooterConfig config;

		runner.expect(
		    !load_from_string(replace(kValidToml, "near_tolerance_rad = 0.08726003490401396",
		                              "near_tolerance_rad = nan"),
		                      config),
		    "NaN near tolerance -> false");
		runner.expect(
		    !load_from_string(replace(kValidToml, "far_tolerance_rad = 0.034904013961605584",
		                              "far_tolerance_rad = inf"),
		                      config),
		    "Inf far tolerance -> false");
		runner.expect(!load_from_string(replace(kValidToml, "distance_threshold_m = 3.0",
		                                        "distance_threshold_m = inf"),
		                                config),
		              "Inf distance threshold -> false");

		runner.end();
	}

	void test_negative_values(TestRunner& runner)
	{
		runner.begin("Negative values");

		auto_aim::ShooterConfig config;

		runner.expect(
		    !load_from_string(replace(kValidToml, "near_tolerance_rad = 0.08726003490401396",
		                              "near_tolerance_rad = -0.1"),
		                      config),
		    "negative near tolerance -> false");
		runner.expect(
		    !load_from_string(replace(kValidToml, "far_tolerance_rad = 0.034904013961605584",
		                              "far_tolerance_rad = -0.1"),
		                      config),
		    "negative far tolerance -> false");
		runner.expect(!load_from_string(replace(kValidToml, "distance_threshold_m = 3.0",
		                                        "distance_threshold_m = -1.0"),
		                                config),
		              "negative distance threshold -> false");

		runner.end();
	}

	void test_near_less_than_far_allowed(TestRunner& runner)
	{
		runner.begin("near < far is allowed (no ordering constraint)");

		auto_aim::ShooterConfig config;

		runner.expect(
		    load_from_string(replace(kValidToml, "near_tolerance_rad = 0.08726003490401396",
		                             "near_tolerance_rad = 0.01"),
		                     config),
		    "near < far loads successfully");

		runner.end();
	}

	void test_compatibility_defaults(TestRunner& runner)
	{
		runner.begin("Compatibility defaults");

		const auto config = auto_aim::make_default_shooter_config();

		runner.expect(!config.auto_fire, "default auto_fire == false (fail-safe)");
		runner.expect(near(config.near_tolerance_rad, 5.0 / 57.3), "default near == 5/57.3");
		runner.expect(near(config.far_tolerance_rad, 2.0 / 57.3), "default far == 2/57.3");
		runner.expect(near(config.distance_threshold_m, 3.0), "default distance == 3.0");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_shooter_config");
	std::printf("=== ShooterConfig Loader Test Suite ===\n\n");

	TestRunner runner;

	test_valid_config(runner);
	test_missing_table_and_field(runner);
	test_invalid_auto_fire_type(runner);
	test_non_finite_values(runner);
	test_negative_values(runner);
	test_near_less_than_far_allowed(runner);
	test_compatibility_defaults(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== ShooterConfig loader tests failed ===\n");
		return 1;
	}

	std::printf("=== All shooter_config tests passed ===\n");
	return 0;
}
