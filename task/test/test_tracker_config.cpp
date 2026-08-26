/**
 * @file test_tracker_config.cpp
 * @brief TrackerConfig TOML loader 单元测试（内存 toml::parse + load_tracker_config_from_table）。
 *
 * 覆盖：
 * - 合法 baseline 配置 -> success
 * - 缺失/非法字段 -> false
 * - 非法 radius profile（非正，或超出 [min_radius_m, max_radius_m]）-> false
 * - 非法 covariance / noise 值 -> false
 * - failed load 不得留下 partially-valid output config
 */

#include "app/auto_aim/tracker_config.hpp"
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

	const char* kValidToml = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [4e-3, 4e-3, 1.0, 9e-2]
distance_angle_log_gain = 1.0
armor_yaw_distance_log_gain = 0.005

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	const char* kMissingField = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [4e-3, 4e-3, 1.0, 9e-2]
distance_angle_log_gain = 1.0
armor_yaw_distance_log_gain = 0.005

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
# max_dt_s missing

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	// radius profile 超出 [min, max]。
	const char* kInvalidRadiusProfile = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [4e-3, 4e-3, 1.0, 9e-2]
distance_angle_log_gain = 1.0
armor_yaw_distance_log_gain = 0.005

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.1

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	// negative variance。
	const char* kInvalidNoise = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [4e-3, 4e-3, 1.0, 9e-2]
distance_angle_log_gain = 1.0
armor_yaw_distance_log_gain = 0.005

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = -1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	// measurement covariance diag 数量错误。
	const char* kInvalidCovariance = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [1e-4, 1e-4, 1e-4]

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	// initial_covariance_diag 出现负值。
	const char* kNegativeInitialCovarianceDiag = R"(
initial_covariance_diag = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [4e-3, 4e-3, 1.0, 9e-2]
distance_angle_log_gain = 1.0
armor_yaw_distance_log_gain = 0.005

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	// measurement_base_covariance_diag 出现负值。
	const char* kNegativeMeasurementCovarianceDiag = R"(
initial_covariance_diag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
measurement_base_covariance_diag = [1e-4, -1e-4, 1e-4, 1e-4]

[lifecycle]
detecting_confirm_hits = 10
detecting_max_misses = 5
temp_lost_max_misses = 20
max_dt_s = 0.5

[association]
max_position_error_m = 0.5
max_yaw_error_rad = 0.5
position_score_scale_m = 1.0
yaw_score_scale_rad = 1.0

[process_noise]
translation_accel_variance = 1.0
yaw_accel_variance = 1.0
radius_random_walk_variance = 1.0
delta_radius_random_walk_variance = 1.0
delta_z_random_walk_variance = 1.0

[radius]
min_radius_m = 0.05
max_radius_m = 0.5

[radius_profile]
balance_2 = 0.2
outpost_3 = 0.2765
base_3 = 0.3205
default_4 = 0.2
)";

	bool load_from_string(const std::string_view toml_text, auto_aim::TrackerConfig& config)
	{
		const toml::table root = toml::parse(toml_text);
		return auto_aim::load_tracker_config_from_table(root, config);
	}

	auto_aim::TrackerConfig make_sentinel_config()
	{
		auto_aim::TrackerConfig c;
		c.detecting_confirm_hits = -1;
		c.max_dt_s = 123.0;
		return c;
	}

	bool is_sentinel_config(const auto_aim::TrackerConfig& config)
	{
		return config.detecting_confirm_hits == -1 && config.max_dt_s == 123.0;
	}

	void test_valid_config(TestRunner& runner)
	{
		runner.begin("Valid baseline config");

		auto_aim::TrackerConfig config;
		const bool ok = load_from_string(kValidToml, config);

		runner.expect(ok, "valid TOML loads successfully");
		runner.expect(config.detecting_confirm_hits == 10, "detecting_confirm_hits == 10");
		runner.expect(config.min_radius_m == 0.05, "min_radius_m == 0.05");
		runner.expect(config.radius_profile.default_4 == 0.2, "default_4 == 0.2");

		// 加载结果必须能通过 validate_tracker_config。
		bool valid = true;
		try
		{
			auto_aim::validate_tracker_config(config);
		}
		catch(...)
		{
			valid = false;
		}
		runner.expect(valid, "loaded config passes validate_tracker_config");

		runner.end();
	}

	void test_missing_or_invalid_field(TestRunner& runner)
	{
		runner.begin("Missing / invalid field");

		auto_aim::TrackerConfig config;
		runner.expect(!load_from_string(kMissingField, config),
		              "missing max_dt_s should fail");

		runner.end();
	}

	void test_invalid_radius_profile(TestRunner& runner)
	{
		runner.begin("Invalid radius profile");

		auto_aim::TrackerConfig config;
		runner.expect(!load_from_string(kInvalidRadiusProfile, config),
		              "radius profile out of [min,max] should fail");

		runner.end();
	}

	void test_invalid_covariance_noise(TestRunner& runner)
	{
		runner.begin("Invalid covariance / noise");

		auto_aim::TrackerConfig config;
		runner.expect(!load_from_string(kInvalidCovariance, config),
		              "measurement covariance wrong size should fail");

		config = auto_aim::TrackerConfig{};
		runner.expect(!load_from_string(kInvalidNoise, config),
		              "negative process noise variance should fail");

		runner.end();
	}

	void test_failed_load_no_partial_output(TestRunner& runner)
	{
		runner.begin("Failed load leaves output unchanged");

		// 对每一种非法输入，output 都必须保持传入前的 sentinel 值。
		const char* invalid_cases[] = {
		    kMissingField, kInvalidRadiusProfile, kInvalidCovariance, kInvalidNoise};

		for(const char* invalid_case: invalid_cases)
		{
			auto_aim::TrackerConfig config = make_sentinel_config();
			const bool ok = load_from_string(invalid_case, config);

			runner.expect(!ok, "invalid config must fail to load");
			runner.expect(is_sentinel_config(config),
			              "failed load must not partially overwrite output");
		}

		runner.end();
	}

	void test_negative_covariance_diag(TestRunner& runner)
	{
		runner.begin("Negative covariance diagonal (TOML)");

		auto_aim::TrackerConfig config;
		runner.expect(!load_from_string(kNegativeInitialCovarianceDiag, config),
		              "negative initial_covariance_diag rejected");
		runner.expect(!load_from_string(kNegativeMeasurementCovarianceDiag, config),
		              "negative measurement_base_covariance_diag rejected");

		runner.end();
	}

	void test_covariance_psd_validation(TestRunner& runner)
	{
		runner.begin("Covariance PSD validation (programmatic)");

		// 合法 baseline。
		auto_aim::TrackerConfig base;
		if(!load_from_string(kValidToml, base))
		{
			runner.expect(false, "baseline config should load");
			runner.end();
			return;
		}

		// 1) symmetric-indefinite：对角 >=0、对称，但存在负特征值 -> 拒绝。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd R =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetMeasurementDim, auto_aim::kTargetMeasurementDim);
			R(0, 0) = 1.0;
			R(0, 1) = 2.0;
			R(1, 0) = 2.0;
			R(1, 1) = 1.0;
			R(2, 2) = 1.0;
			R(3, 3) = 1.0;
			c.measurement_noise.base_covariance = R;

			bool rejected = false;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(const std::invalid_argument&)
			{
				rejected = true;
			}
			runner.expect(rejected, "symmetric-indefinite measurement_covariance rejected");
		}

		// 2) programmatic 负对角 initial covariance -> 拒绝。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd P =
			    Eigen::MatrixXd::Identity(auto_aim::kTargetStateDim, auto_aim::kTargetStateDim);
			P(0, 0) = -1.0;
			c.initial_covariance = P;

			bool rejected = false;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(const std::invalid_argument&)
			{
				rejected = true;
			}
			runner.expect(rejected, "negative diagonal initial_covariance rejected");
		}

		// 3) zero covariance -> 接受。
		{
			auto_aim::TrackerConfig c = base;
			c.initial_covariance =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetStateDim, auto_aim::kTargetStateDim);
			c.measurement_noise.base_covariance =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetMeasurementDim, auto_aim::kTargetMeasurementDim);

			bool accepted = true;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(...)
			{
				accepted = false;
			}
			runner.expect(accepted, "zero covariance accepted");
		}

		// 4) singular PSD covariance -> 接受。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd P =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetStateDim, auto_aim::kTargetStateDim);
			P(0, 0) = 1.0; // 仅一个非零对角 -> singular PSD
			c.initial_covariance = P;

			bool accepted = true;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(...)
			{
				accepted = false;
			}
			runner.expect(accepted, "singular PSD initial_covariance accepted");
		}

		// 5) normal PSD covariance -> 接受。
		{
			bool accepted = true;
			try
			{
				auto_aim::validate_tracker_config(base);
			}
			catch(...)
			{
				accepted = false;
			}
			runner.expect(accepted, "normal PSD covariance accepted");
		}

		// 6) small-but-real negative eigenvalue -> 拒绝。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd R =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetMeasurementDim, auto_aim::kTargetMeasurementDim);
			R(0, 0) = 1.0;
			R(1, 1) = 1.0;
			R(2, 2) = 1.0;
			R(3, 3) = 1.0;
			// 2x2 块 [[1, 1+1e-10],[1+1e-10, 1]] -> λ_min = -1e-10（对角仍 >= 0）
			R(0, 1) = 1.0 + 1e-10;
			R(1, 0) = 1.0 + 1e-10;
			c.measurement_noise.base_covariance = R;

			bool rejected = false;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(const std::invalid_argument&)
			{
				rejected = true;
			}
			runner.expect(rejected, "small-but-real negative eigenvalue rejected");
		}

		// 7) roundoff 级别负特征值 -> 接受。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd R =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetMeasurementDim, auto_aim::kTargetMeasurementDim);
			R(0, 0) = 1.0;
			R(1, 1) = 1.0;
			R(2, 2) = 1.0;
			R(3, 3) = 1.0;
			// 2x2 块 [[1, 1+1e-15],[1+1e-15, 1]] -> λ_min = -1e-15（roundoff 级别）
			R(0, 1) = 1.0 + 1e-15;
			R(1, 0) = 1.0 + 1e-15;
			c.measurement_noise.base_covariance = R;

			bool accepted = true;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(...)
			{
				accepted = false;
			}
			runner.expect(accepted, "roundoff-level negative eigenvalue accepted");
		}

		// 8) large-scale indefinite -> 拒绝。
		{
			auto_aim::TrackerConfig c = base;
			Eigen::MatrixXd R =
			    Eigen::MatrixXd::Zero(auto_aim::kTargetMeasurementDim, auto_aim::kTargetMeasurementDim);
			R(0, 0) = 100.0;
			R(1, 1) = 100.0;
			R(2, 2) = 1.0;
			R(3, 3) = 1.0;
			// 2x2 块 [[100, 200],[200, 100]] -> λ = {300, -100}
			R(0, 1) = 200.0;
			R(1, 0) = 200.0;
			c.measurement_noise.base_covariance = R;

			bool rejected = false;
			try
			{
				auto_aim::validate_tracker_config(c);
			}
			catch(const std::invalid_argument&)
			{
				rejected = true;
			}
			runner.expect(rejected, "large-scale indefinite covariance rejected");
		}

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_tracker_config");
	std::printf("=== TrackerConfig Loader Test Suite ===\n\n");

	TestRunner runner;

	test_valid_config(runner);
	test_missing_or_invalid_field(runner);
	test_invalid_radius_profile(runner);
	test_invalid_covariance_noise(runner);
	test_negative_covariance_diag(runner);
	test_covariance_psd_validation(runner);
	test_failed_load_no_partial_output(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== TrackerConfig loader tests failed ===\n");
		return 1;
	}

	std::printf("=== All tracker_config tests passed ===\n");
	return 0;
}