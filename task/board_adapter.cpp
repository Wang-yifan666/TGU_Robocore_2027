/**
 * @file board_adapter.cpp
 * @brief task 层板卡数据适配实现。
 */

#include "board_adapter.hpp"

#include <cmath>

#include "tools/maths_tools.hpp"

namespace task
{

    double gimbal_world_yaw_rad(const Eigen::Quaterniond& q_imu_body_to_world,
                                const Eigen::Matrix3d& r_gimbal_to_imu_body)
    {
        const Eigen::Matrix3d r_imu_body_to_world =
            q_imu_body_to_world.normalized().toRotationMatrix();

        const Eigen::Matrix3d r_gimbal_to_world =
            r_gimbal_to_imu_body.transpose() * r_imu_body_to_world * r_gimbal_to_imu_body;

        return tools::maths_tools::limit_rad(
            tools::maths_tools::rot_to_euler(r_gimbal_to_world, 2, 1, 0, true)[0]);
    }

    app::auto_aim::FrameContext make_frame_context(const io::CameraFrame& camera,
                                                   const BoardFeedback& board,
                                                   const Eigen::Matrix3d& r_gimbal_to_imu_body)
    {
        app::auto_aim::FrameContext frame;

        frame.image = camera.image;
        frame.timestamp_s = camera.timestamp_s;

        frame.q_imu_body_to_world = board.has_quaternion
                                        ? board.q_imu_body_to_world.normalized()
                                        : Eigen::Quaterniond::Identity();

        const bool bullet_valid = board.has_bullet_speed && std::isfinite(board.bullet_speed_mps)
            && board.bullet_speed_mps > 0.0;
        frame.bullet_speed_mps = bullet_valid ? board.bullet_speed_mps
                                              : std::numeric_limits<double>::quiet_NaN();

        if(board.has_gimbal_yaw && std::isfinite(board.gimbal_yaw_rad))
        {
            frame.gimbal_yaw_rad = tools::maths_tools::limit_rad(board.gimbal_yaw_rad);
        }
        else if(board.has_quaternion)
        {
            frame.gimbal_yaw_rad =
                gimbal_world_yaw_rad(board.q_imu_body_to_world, r_gimbal_to_imu_body);
        }
        else
        {
            frame.gimbal_yaw_rad = std::numeric_limits<double>::quiet_NaN();
        }

        return frame;
    }

    GimbalCommand make_gimbal_command(const app::auto_aim::AimResult& result)
    {
        GimbalCommand command; // 默认 control=false, yaw/pitch=0, fire=false

        if(result.has_aim)
        {
            command.control = true;
            command.yaw_rad = result.yaw;
            command.pitch_rad = result.pitch;
            command.fire = result.fire;
        }

        return command;
    }

} // namespace task
