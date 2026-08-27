/**
 * @file board_adapter.hpp
 * @brief task 层板卡数据适配（app 与 io 之间，禁止协议类型进入 app）。
 *
 * 方向：
 *   io  -> task adapter -> FrameContext -> AutoAim
 *   AimResult -> task adapter -> GimbalCommand -> io
 */

#ifndef TGU_ROBOCORE_2027_TASK_BOARD_ADAPTER_HPP
#define TGU_ROBOCORE_2027_TASK_BOARD_ADAPTER_HPP

#include <limits>
#include <optional>

#include <Eigen/Geometry>

#include "app/auto_aim/auto_aim.hpp"
#include "io/usbcamera/usbcamera.hpp"

namespace task
{

	/**
     * @brief 板卡反馈（task 层抽象，与具体串口/CAN 协议无关）。
     *
     * 姿态真源唯一：q_imu_body_to_world（IMU body -> world）。
     * gimbal yaw 一律由 quaternion + hand-eye transform 派生，不另设直给 yaw。
     */
	struct BoardFeedback
	{
		bool has_quaternion = false;
		Eigen::Quaterniond q_imu_body_to_world = Eigen::Quaterniond::Identity();

		bool has_bullet_speed = false;
		double bullet_speed_mps = std::numeric_limits<double>::quiet_NaN();
	};

	/**
     * @brief 云台控制命令（task 层抽象，rad，absolute world yaw）。
     *
     * 每帧由 make_gimbal_command 全量重建，避免 stale fire/command。
     */
	struct GimbalCommand
	{
		bool control = false;
		double yaw_rad = 0.0;
		double pitch_rad = 0.0;
		bool fire = false;
	};

	/**
     * @brief 校验 quaternion 是否为有效姿态（系数 finite 且 norm 非零）。
     */
	bool is_valid_quaternion(const Eigen::Quaterniond& quat);

	/**
     * @brief 由 IMU 四元数 + 固定外参派生 gimbal 世界系 yaw。
     *
     * 使用与 Solver 完全相同的 Euler convention：
     * rot_to_euler(R, 2, 1, 0)（默认 is_extrinsic=false）。
     */
	double gimbal_world_yaw_rad(const Eigen::Quaterniond& q_imu_body_to_world,
	                            const Eigen::Matrix3d& r_gimbal_to_imu_body);

	/**
     * @brief board feedback -> FrameContext。
     *
     * 无有效姿态（quaternion missing / NaN / zero-norm）时返回 nullopt，
     * 调用方必须 fail closed（不产生 control/fire）。
     */
	std::optional<app::auto_aim::FrameContext> make_frame_context(
	    const io::CameraFrame& camera, const BoardFeedback& board,
	    const Eigen::Matrix3d& r_gimbal_to_imu_body);

	/**
     * @brief AimResult -> GimbalCommand（fail-safe）。
     *
     * control = has_aim && finite(yaw) && finite(pitch)；
     * fire = result.fire && control。
     */
	GimbalCommand make_gimbal_command(const app::auto_aim::AimResult& result);

} // namespace task

#endif // TGU_ROBOCORE_2027_TASK_BOARD_ADAPTER_HPP
