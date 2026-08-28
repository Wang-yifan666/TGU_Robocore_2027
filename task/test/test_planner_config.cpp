/**
 * @file test_planner_config.cpp
 * @brief PlannerConfig 校验与 TOML 加载单元测试。
 */

#include "app/auto_aim/planner_config.hpp"

#include <cstdio>
#include <stdexcept>
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

	namespace auto_aim = app::auto_aim;

	bool parse(const char* toml_text, auto_aim::PlannerConfig& config)
	{
		const toml::table root = toml::parse(toml_text);
		return auto_aim::load_planner_config_from_table(root, config);
	}

	void test_validation(TestRunner& runner)
	{
		runner.begin("validate_planner_config");

		const auto throws = [](const auto_aim::PlannerConfig& c) {
			try
			{
				auto_aim::validate_planner_config(c);
				return false;
			}
			catch(const std::invalid_argument&)
			{
				return true;
			}
		};

		runner.expect(!throws(auto_aim::make_default_planner_config()), "default config valid");

		{
			auto c = auto_aim::make_default_planner_config();
			c.q_yaw = Eigen::Vector2d(9e6, 0.0);
			runner.expect(!throws(c), "Q = [9e6, 0] valid (positive semidefinite)");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.q_pitch = Eigen::Vector2d(9e6, -1.0);
			runner.expect(throws(c), "negative Q element throws");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.r_yaw = 0.0;
			runner.expect(throws(c), "r == 0 throws");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.rho = 0.0;
			runner.expect(throws(c), "rho == 0 throws");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.max_yaw_acceleration_rad_s2 = 0.0;
			runner.expect(throws(c), "max_yaw_acc == 0 throws");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.max_pitch_acceleration_rad_s2 = -1.0;
			runner.expect(throws(c), "max_pitch_acc < 0 throws");
		}
		{
			auto c = auto_aim::make_default_planner_config();
			c.max_iter = 0;
			runner.expect(throws(c), "max_iter < 1 throws");
		}

		runner.end();
	}

	void test_load(TestRunner& runner)
	{
		runner.begin("load_planner_config_from_table");

		auto_aim::PlannerConfig config;

		const char* valid = R"(
[planner]
max_yaw_acceleration_rad_s2 = 50.0
max_pitch_acceleration_rad_s2 = 100.0
q_yaw = [9e6, 0.0]
r_yaw = 1.0
q_pitch = [9e6, 0.0]
r_pitch = 1.0
rho = 1.0
max_iter = 10
)";
		runner.expect(parse(valid, config), "valid config loads");
		runner.expect(config.q_yaw(0) == 9e6 && config.q_yaw(1) == 0.0, "q_yaw parsed");
		runner.expect(config.max_iter == 10, "max_iter parsed");

		runner.expect(!parse("[other]\nx = 1\n", config), "missing [planner] table fails");
		runner.expect(!parse("[planner]\nmax_yaw_acceleration_rad_s2 = 50.0\n", config),
		              "missing field fails");
		runner.expect(!parse("[planner]\nmax_yaw_acceleration_rad_s2 = 50.0\nmax_pitch_acceleration_rad_s2 = 100.0\nq_yaw = [1.0]\nr_yaw = 1.0\nq_pitch = [1.0, 0.0]\nr_pitch = 1.0\nrho = 1.0\nmax_iter = 10\n", config),
		              "q_yaw wrong size fails");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_planner_config");
	std::printf("=== PlannerConfig Test Suite ===\n\n");

	TestRunner runner;
	test_validation(runner);
	test_load(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== PlannerConfig tests failed ===\n");
		return 1;
	}

	std::printf("=== All PlannerConfig tests passed ===\n");
	return 0;
}
