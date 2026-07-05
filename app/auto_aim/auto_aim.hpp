//
// Created by tgu on 2026/4/29.
//

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_HPP

#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "app/auto_aim/armor.hpp"
#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	struct AutoAimConfig
	{
		ArmorColor enemy_color = ArmorColor::Blue;

		bool enable_detector = false;
		bool enable_solver = false;
		bool enable_tracker = false;
		bool enable_predictor = false;
		bool enable_debug = true;
	};

	struct FrameContext
	{
		cv::Mat image;
		double timestamp_s = 0.0;

		// 后续接 IMU 时使用
		Eigen::Quaterniond q_imu_body_to_world = Eigen::Quaterniond::Identity();
	};

	enum class AimState : std::uint8_t
	{
		Idle = 0,
		NoFrame,
		NoTarget,
		Detecting,
		Tracking,
		TargetLocked,
		Error
	};

	struct AimResult
	{
		bool has_target = false;

		AimState state = AimState::Idle;

		Armor target;

		double yaw = 0.0;
		double pitch = 0.0;
		double distance = 0.0;

		double timestamp_s = 0.0;
	};

	class AutoAim
	{
	public:
		AutoAim() = default;
		~AutoAim() = default;

		bool init(const AutoAimConfig& config);

		void reset();

		AimResult process(const FrameContext& frame);

		bool is_ready() const noexcept;

	private:
		AutoAimConfig config_;
		bool initialized_ = false;
		std::uint64_t frame_count_ = 0;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_HPP
