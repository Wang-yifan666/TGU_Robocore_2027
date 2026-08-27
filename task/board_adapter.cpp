/**
 * @file board_adapter.cpp
 * @brief task 层板卡数据适配实现。
 */

#include "board_adapter.hpp"

#include <cmath>

#include "tools/maths_tools.hpp"

namespace task
{

	bool is_valid_quaternion(const Eigen::Quaterniond& quat)
	{
		const Eigen::Vector4d& coeffs = quat.coeffs();

		for(int i = 0; i < 4; ++i)
		{
			if(!std::isfinite(coeffs[i]))
			{
				return false;
			}
		}

		return quat.squaredNorm() > 1e-12;
	}

	double gimbal_world_yaw_rad(const Eigen::Quaterniond& q_imu_body_to_world,
	                            const Eigen::Matrix3d& r_gimbal_to_imu_body)
	{
		const Eigen::Matrix3d r_imu_body_to_world =
		    q_imu_body_to_world.normalized().toRotationMatrix();

		const Eigen::Matrix3d r_gimbal_to_world =
		    r_gimbal_to_imu_body.transpose() * r_imu_body_to_world * r_gimbal_to_imu_body;

		// 与 Solver 相同的 Euler convention：rot_to_euler(R, 2, 1, 0) 默认 is_extrinsic=false。
		return tools::maths_tools::limit_rad(
		    tools::maths_tools::rot_to_euler(r_gimbal_to_world, 2, 1, 0)[0]);
	}

	std::optional<app::auto_aim::FrameContext> make_frame_context(
	    const io::CameraFrame& camera, const BoardFeedback& board,
	    const Eigen::Matrix3d& r_gimbal_to_imu_body)
	{
		// 无有效姿态：fail closed，不产出可用于控制的 FrameContext。
		if(!board.has_quaternion || !is_valid_quaternion(board.q_imu_body_to_world))
		{
			return std::nullopt;
		}

		app::auto_aim::FrameContext frame;

		frame.image = camera.image;
		frame.timestamp_s = camera.timestamp_s;
		frame.q_imu_body_to_world = board.q_imu_body_to_world.normalized();

		const bool bullet_valid = board.has_bullet_speed && std::isfinite(board.bullet_speed_mps)
		    && board.bullet_speed_mps > 0.0;
		frame.bullet_speed_mps =
		    bullet_valid ? board.bullet_speed_mps : std::numeric_limits<double>::quiet_NaN();

		frame.gimbal_yaw_rad = gimbal_world_yaw_rad(board.q_imu_body_to_world, r_gimbal_to_imu_body);

		return frame;
	}

	GimbalCommand make_gimbal_command(const app::auto_aim::AimResult& result)
	{
		GimbalCommand command; // 默认 control=false, yaw/pitch=0, fire=false

		const bool aim_valid =
		    result.has_aim && std::isfinite(result.yaw) && std::isfinite(result.pitch);

		if(aim_valid)
		{
			command.control = true;
			command.yaw_rad = result.yaw;
			command.pitch_rad = result.pitch;
		}

		command.fire = result.fire && command.control;

		return command;
	}

} // namespace task
