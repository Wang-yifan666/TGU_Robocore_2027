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

#include <Eigen/Geometry>

#include "app/auto_aim/auto_aim.hpp"
#include "io/usbcamera/usbcamera.hpp"

namespace task
{

    /**
     * @brief 板卡反馈（task 层抽象，与具体串口/CAN 协议无关）。
     *
     * 单位约定：角度 rad，速度 m/s。无效/缺失用 has_* 标志区分，
     * 数值缺失时保持 NaN，不伪造 0。
     */
    struct BoardFeedback
    {
        bool has_quaternion = false;
        Eigen::Quaterniond q_imu_body_to_world = Eigen::Quaterniond::Identity();

        bool has_bullet_speed = false;
        double bullet_speed_mps = std::numeric_limits<double>::quiet_NaN();

        bool has_gimbal_yaw = false;
        double gimbal_yaw_rad = std::numeric_limits<double>::quiet_NaN();
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
     * @brief 由 IMU 四元数 + 固定外参派生 gimbal 世界系 yaw。
     *
     * R_gw = R_gimbal_to_imu_body^T * R_imu_body_to_world * R_gimbal_to_imu_body，
     * yaw 取 R_gw 的 ZYX 外旋 yaw，并 wrap 到 [-pi, pi)。
     */
    double gimbal_world_yaw_rad(const Eigen::Quaterniond& q_imu_body_to_world,
                                const Eigen::Matrix3d& r_gimbal_to_imu_body);

    /**
     * @brief board feedback -> FrameContext。
     *
     * - image/timestamp 取自相机帧；
     * - 弹速非法/缺失 -> NaN（由 Aimer policy 决定 fallback/fail-safe）；
     * - gimbal yaw 优先用板端直给值，否则由四元数派生，否则 NaN。
     */
    app::auto_aim::FrameContext make_frame_context(
        const io::CameraFrame& camera, const BoardFeedback& board,
        const Eigen::Matrix3d& r_gimbal_to_imu_body);

    /**
     * @brief AimResult -> GimbalCommand（fail-safe）。
     *
     * control = has_aim；fire = result.fire（has_aim 为 false 时必为 false）。
     * has_aim 为 false 时 yaw/pitch 归零，不残留。
     */
    GimbalCommand make_gimbal_command(const app::auto_aim::AimResult& result);

} // namespace task

#endif // TGU_ROBOCORE_2027_TASK_BOARD_ADAPTER_HPP
