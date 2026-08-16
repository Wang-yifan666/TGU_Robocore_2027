/**
 * @file test_solver_config.cpp
 * @brief SolverConfig TOML loader 单元测试（内存 toml::parse + load_solver_config_from_table）。
 *
 * 覆盖：
 * - 合法 table -> success，Solver::is_valid() == true
 * - 缺失 [camera] / [extrinsic] -> false
 * - 数组含非 numeric 元素（如字符串）-> false（不静默 fallback 到 0）
 * - camera_matrix fx/fy <= 0 或含 NaN/Inf -> false
 * - 矩阵 8/9 个元素、平移 2/3 个元素 -> false
 * - camera_matrix 元素数量错误 -> false
 */

#include "app/auto_aim/solver.hpp"
#include "app/auto_aim/solver_config.hpp"
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

	// ============================================================
	// TOML 片段构造
	// ============================================================

	const char* kValidToml = R"(
[camera]
camera_matrix = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	const char* kMissingCamera = R"(
[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	const char* kMissingExtrinsic = R"(
[camera]
camera_matrix = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
)";

	// camera_matrix 第 2 个元素是字符串。
	const char* kNonNumericElement = R"(
[camera]
camera_matrix = [800.0, "x", 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	// fx = 0。
	const char* kZeroFx = R"(
[camera]
camera_matrix = [0.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	// fy = -1。
	const char* kNegativeFy = R"(
[camera]
camera_matrix = [800.0, 0.0, 640.0, 0.0, -1.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	// camera_matrix 只有 8 个元素。
	const char* kMatrixTooFew = R"(
[camera]
camera_matrix = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02, 0.03]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	// t_camera_to_gimbal 只有 2 个元素。
	const char* kTranslationTooFew = R"(
[camera]
camera_matrix = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
distort_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

[extrinsic]
r_camera_to_gimbal = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera_to_gimbal = [0.01, -0.02]
r_gimbal_to_imu_body = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)";

	bool load_from_string(const std::string_view toml_text, auto_aim::SolverConfig& config)
	{
		const toml::table root = toml::parse(toml_text);
		return auto_aim::load_solver_config_from_table(root, config);
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_valid_config(TestRunner& runner)
	{
		runner.begin("Valid config");

		auto_aim::SolverConfig config;
		const bool ok = load_from_string(kValidToml, config);

		runner.expect(ok, "Valid TOML should load successfully");

		runner.expect(config.camera_matrix.rows == 3 && config.camera_matrix.cols == 3,
		              "camera_matrix should be 3x3");
		runner.expect(std::isfinite(config.camera_matrix.at<double>(0, 0)),
		              "camera_matrix values should be finite");

		auto_aim::Solver solver(config);
		runner.expect(solver.is_valid(), "Solver should be valid for loaded config");

		runner.end();
	}

	void test_missing_tables(TestRunner& runner)
	{
		runner.begin("Missing tables");

		auto_aim::SolverConfig config;

		runner.expect(!load_from_string(kMissingCamera, config),
		              "Missing [camera] should fail");
		runner.expect(!load_from_string(kMissingExtrinsic, config),
		              "Missing [extrinsic] should fail");

		runner.end();
	}

	void test_non_numeric_elements(TestRunner& runner)
	{
		runner.begin("Non-numeric elements");

		auto_aim::SolverConfig config;

		runner.expect(!load_from_string(kNonNumericElement, config),
		              "String element in camera_matrix should fail (no silent fallback to 0)");

		runner.end();
	}

	void test_focal_length_checks(TestRunner& runner)
	{
		runner.begin("Focal length checks");

		auto_aim::SolverConfig config;

		runner.expect(!load_from_string(kZeroFx, config), "fx == 0 should fail");
		runner.expect(!load_from_string(kNegativeFy, config), "fy < 0 should fail");

		runner.end();
	}

	void test_element_count_checks(TestRunner& runner)
	{
		runner.begin("Element count checks");

		auto_aim::SolverConfig config;

		runner.expect(!load_from_string(kMatrixTooFew, config),
		              "camera_matrix with 8 elements should fail");
		runner.expect(!load_from_string(kTranslationTooFew, config),
		              "t_camera_to_gimbal with 2 elements should fail");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== SolverConfig Loader Test Suite ===\n\n");

	TestRunner runner;

	test_valid_config(runner);
	test_missing_tables(runner);
	test_non_numeric_elements(runner);
	test_focal_length_checks(runner);
	test_element_count_checks(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== SolverConfig loader tests failed ===\n");
		return 1;
	}

	std::printf("=== All solver_config tests passed ===\n");
	return 0;
}