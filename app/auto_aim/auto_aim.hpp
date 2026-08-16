/**
 * @file auto_aim.hpp
 * @brief 自瞄算法 facade / orchestrator（pre-tracker 阶段）。
 *
 * 本阶段职责：
 *   Detector -> 预跟踪确定性目标选择 -> Solver -> AimResult
 *
 * 不负责：
 * - 读取 TOML 配置；
 * - 构造 OpenVINOInference；
 * - 访问 camera / serial。
 *
 * 依赖装配（Detector / Solver）由 task 层完成并注入。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_HPP

#pragma once

#include <cstdint>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "app/auto_aim/armor.hpp"
#include "app/auto_aim/detector/detector.hpp"
#include "app/auto_aim/solver.hpp"
#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 单帧输入上下文。
	 */
	struct FrameContext
	{
		cv::Mat image;
		double timestamp_s = 0.0;

		// 云台 body -> 世界 四元数。
		// 无同步 IMU 数据时必须保持 Identity（仅离线验证）。
		Eigen::Quaterniond q_imu_body_to_world = Eigen::Quaterniond::Identity();
	};

	/**
	 * @brief 自瞄状态。
	 *
	 * Tracking / TargetLocked 保留给后续 Tracker / Aimer 阶段，
	 * 本阶段不产生这两个状态。
	 */
	enum class AimState : std::uint8_t
	{
		Idle = 0,
		NoFrame,
		NoTarget,
		Detecting,
		Tracking,     // 保留，后续 Tracker 阶段使用
		TargetLocked, // 保留，后续 Tracker/Aimer 阶段使用
		Error
	};

	/**
	 * @brief 单帧自瞄输出。
	 */
	struct AimResult
	{
		bool has_target = false;

		AimState state = AimState::Idle;

		Armor target;

		/**
		 * @brief raw geometric line-of-sight observation，NOT final ballistic compensated command。
		 *
		 * 本阶段不定义云台控制命令语义，因此 yaw / pitch 保持默认 0；
		 * 主要结果见 target.xyz_in_gimbal / target.xyz_in_world / target.ypr / target.ypd。
		 */
		double yaw = 0.0;
		double pitch = 0.0;

		/**
		 * @brief 目标在云台坐标系下的距离（单位 m，= |xyz_in_gimbal|）。
		 */
		double distance = 0.0;

		double timestamp_s = 0.0;
	};

	/**
	 * @brief 自瞄 facade：持有 Detector 与 Solver，逐帧编排检测/选择/解算。
	 */
	class AutoAim
	{
	public:
		AutoAim(Detector detector, Solver solver);

		AimResult process(const FrameContext& frame);

		void reset();

		bool is_ready() const noexcept;

	private:
		Detector detector_;
		Solver solver_;

		std::uint64_t frame_count_ = 0;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_HPP