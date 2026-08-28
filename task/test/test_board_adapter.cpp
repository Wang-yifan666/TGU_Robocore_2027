/**
 * @file test_board_adapter.cpp
 * @brief task 层 board adapter 单元测试（input/output 映射，不依赖硬件）。
 */

#include "task/board_adapter.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

#include <Eigen/Geometry>

#include "test_logging.hpp"
#include "tools/maths_tools.hpp"

namespace
{

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

	io::CameraFrame make_camera_frame()
	{
		io::CameraFrame camera;
		camera.timestamp_s = 1.0;
		return camera;
	}

	task::BoardFeedback make_feedback(bool has_quat = true, bool has_bullet = false,
	                                  double bullet = 0.0)
	{
		task::BoardFeedback board;
		board.has_quaternion = has_quat;
		board.q_imu_body_to_world = Eigen::Quaterniond::Identity();
		board.has_bullet_speed = has_bullet;
		board.bullet_speed_mps = bullet;
		return board;
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_bullet_speed_mapping(TestRunner& runner)
	{
		runner.begin("bullet speed mapping");

		const double nan = std::numeric_limits<double>::quiet_NaN();
		const io::CameraFrame camera = make_camera_frame();
		const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

		{
			task::BoardFeedback board = make_feedback(true, true, 23.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(frame.has_value(), "valid feedback -> frame");
			runner.expect(near(frame->bullet_speed_mps, 23.0), "valid 23 m/s passed through");
		}
		{
			task::BoardFeedback board = make_feedback(true, true, 0.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(std::isnan(frame->bullet_speed_mps), "zero -> NaN");
		}
		{
			task::BoardFeedback board = make_feedback(true, true, -5.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(std::isnan(frame->bullet_speed_mps), "negative -> NaN");
		}
		{
			task::BoardFeedback board = make_feedback(true, true, nan);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(std::isnan(frame->bullet_speed_mps), "NaN -> NaN");
		}
		{
			task::BoardFeedback board = make_feedback(true, false, 0.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(std::isnan(frame->bullet_speed_mps), "missing -> NaN");
		}

		runner.end();
	}

	void test_gimbal_yaw_derivation(TestRunner& runner)
	{
		runner.begin("gimbal yaw derivation");

		const io::CameraFrame camera = make_camera_frame();
		const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
		const double theta = 0.3;

		{
			task::BoardFeedback board = make_feedback(true);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(near(frame->gimbal_yaw_rad, 0.0), "identity -> 0");
		}
		{
			task::BoardFeedback board = make_feedback(true);
			board.q_imu_body_to_world = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(near(frame->gimbal_yaw_rad, theta), "Rz(theta) -> theta");
		}
		{
			Eigen::Matrix3d r_gi = Eigen::Matrix3d::Zero();
			r_gi(0, 0) = -1.0;
			r_gi(1, 1) = -1.0;
			r_gi(2, 2) = 1.0;

			task::BoardFeedback board = make_feedback(true);
			board.q_imu_body_to_world = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
			const auto frame = task::make_frame_context(camera, board, r_gi);
			runner.expect(near(frame->gimbal_yaw_rad, theta), "r_gi=180z + Rz(theta) -> theta");
		}

		runner.end();
	}

	void test_gimbal_yaw_combined_pose(TestRunner& runner)
	{
		runner.begin("gimbal yaw combined yaw+pitch+roll");

		const io::CameraFrame camera = make_camera_frame();
		const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

		const double yaw = 0.4;
		const double pitch = 0.3;
		const double roll = -0.2;

		const Eigen::Matrix3d r_iw =
		    tools::maths_tools::rotation_matrix(Eigen::Vector3d{yaw, pitch, roll});
		const Eigen::Quaterniond q(r_iw);

		task::BoardFeedback board = make_feedback(true);
		board.q_imu_body_to_world = q;

		const auto frame = task::make_frame_context(camera, board, identity);
		runner.expect(frame.has_value(), "combined pose -> frame");
		runner.expect(near(frame->gimbal_yaw_rad, yaw), "combined pose yaw extracted");

		runner.end();
	}

	void test_invalid_quaternion(TestRunner& runner)
	{
		runner.begin("invalid quaternion -> no frame");

		const double nan = std::numeric_limits<double>::quiet_NaN();
		const io::CameraFrame camera = make_camera_frame();
		const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

		{
			task::BoardFeedback board; // has_quaternion = false
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(!frame.has_value(), "missing quaternion -> nullopt");
		}
		{
			task::BoardFeedback board = make_feedback(true);
			board.q_imu_body_to_world = Eigen::Quaterniond(nan, 0.0, 0.0, 0.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(!frame.has_value(), "NaN quaternion -> nullopt");
		}
		{
			task::BoardFeedback board = make_feedback(true);
			board.q_imu_body_to_world = Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0);
			const auto frame = task::make_frame_context(camera, board, identity);
			runner.expect(!frame.has_value(), "zero-norm quaternion -> nullopt");
		}

		runner.end();
	}

	void test_command_mapping(TestRunner& runner)
	{
		runner.begin("command mapping");

		const double nan = std::numeric_limits<double>::quiet_NaN();

		{
			app::auto_aim::AimResult result;
			result.has_plan = true;
			result.fire = false;
			result.yaw = 0.5;
			result.pitch = -0.3;
			const auto command = task::make_gimbal_command(result);
			runner.expect(command.control, "control true");
			runner.expect(near(command.yaw_rad, 0.5) && near(command.pitch_rad, -0.3),
			              "yaw/pitch copied");
			runner.expect(!command.fire, "fire false");
		}
		{
			app::auto_aim::AimResult result;
			result.has_plan = true;
			result.fire = true;
			result.yaw = 0.5;
			result.pitch = -0.3;
			const auto command = task::make_gimbal_command(result);
			runner.expect(command.fire, "fire true");
		}
		{
			app::auto_aim::AimResult result;
			result.has_plan = false;
			result.fire = true; // 非法输入，仍须 fail-safe
			result.yaw = nan;
			result.pitch = nan;
			const auto command = task::make_gimbal_command(result);
			runner.expect(!command.control && !command.fire, "control/fire false");
			runner.expect(command.yaw_rad == 0.0 && command.pitch_rad == 0.0, "yaw/pitch zero");
		}
		{
			// has_aim=true 但 has_plan=false：即便 Shooter 单独判 true，也不允许 control/fire。
			app::auto_aim::AimResult result;
			result.has_aim = true;
			result.has_plan = false;
			result.fire = true;
			result.yaw = 0.5;
			result.pitch = -0.3;
			const auto command = task::make_gimbal_command(result);
			runner.expect(!command.control && !command.fire,
			              "has_plan=false -> control/fire false (even if has_aim=true)");
		}
		{
			app::auto_aim::AimResult result;
			result.has_plan = true;
			result.fire = true;
			result.yaw = nan;
			result.pitch = -0.3;
			const auto command = task::make_gimbal_command(result);
			runner.expect(!command.control, "NaN yaw -> control false");
			runner.expect(!command.fire, "NaN yaw -> fire false");
		}
		{
			app::auto_aim::AimResult result;
			result.has_plan = true;
			result.fire = true;
			result.yaw = 0.5;
			result.pitch = nan;
			const auto command = task::make_gimbal_command(result);
			runner.expect(!command.control, "NaN pitch -> control false");
			runner.expect(!command.fire, "NaN pitch -> fire false");
		}
		{
			app::auto_aim::AimResult first;
			first.has_plan = true;
			first.fire = true;
			first.yaw = 0.5;
			first.pitch = -0.3;
			task::GimbalCommand command = task::make_gimbal_command(first);
			runner.expect(command.fire, "first fire true");

			app::auto_aim::AimResult second;
			second.has_plan = false;
			second.fire = false;
			second.yaw = nan;
			second.pitch = nan;
			command = task::make_gimbal_command(second);
			runner.expect(!command.fire, "no stale fire after reuse");
			runner.expect(!command.control, "no stale control after reuse");
		}

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_board_adapter");
	std::printf("=== Board Adapter Test Suite ===\n\n");

	TestRunner runner;

	test_bullet_speed_mapping(runner);
	test_gimbal_yaw_derivation(runner);
	test_gimbal_yaw_combined_pose(runner);
	test_invalid_quaternion(runner);
	test_command_mapping(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Board adapter tests failed ===\n");
		return 1;
	}

	std::printf("=== All board adapter tests passed ===\n");
	return 0;
}
