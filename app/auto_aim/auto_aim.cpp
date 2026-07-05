//
// Created by tgu on 2026/4/29.
//

#include "app/auto_aim/auto_aim.hpp"

#include <vector>

#include "tools/logger.hpp"

namespace app::auto_aim
{

	namespace
	{
		static constexpr const char* MODULE = "AUTO_AIM";
	}

	bool AutoAim::init(const AutoAimConfig& config)
	{
		config_ = config;
		initialized_ = true;
		frame_count_ = 0;

		LOG_INFO(MODULE, "auto aim initialized, detector={}, solver={}, tracker={}, predictor={}",
		         config_.enable_detector, config_.enable_solver, config_.enable_tracker,
		         config_.enable_predictor);

		return true;
	}

	void AutoAim::reset()
	{
		frame_count_ = 0;
		LOG_INFO(MODULE, "auto aim reset");
	}

	bool AutoAim::is_ready() const noexcept
	{
		return initialized_;
	}

	AimResult AutoAim::process(const FrameContext& frame)
	{
		AimResult result;
		result.timestamp_s = frame.timestamp_s;

		if(!initialized_)
		{
			result.state = AimState::Error;
			LOG_ERROR(MODULE, "auto aim is not initialized");
			return result;
		}

		if(frame.image.empty())
		{
			result.state = AimState::NoFrame;
			LOG_WARN(MODULE, "input frame is empty");
			return result;
		}

		++ frame_count_;

		// TODO : detector
		// 调用推理
		// 输出 std::vector<Armor> armors
		std::vector<Armor> armors;

		// TODO : armor
		// 过滤

		if(armors.empty())
		{
			result.state = AimState::NoTarget;
			LOG_DEBUG(MODULE, "frame {}: no armor detected", frame_count_);
			return result;
		}

		// TODO : solver
		// 对候选装甲板做 PnP，得到 xyz / yaw / pitch / distance

		// TODO : tracker
		// 跟踪，避免目标跳变

		// TODO : predictor
		// 弹道补偿、延迟补偿、是否开火

		result.state = AimState::NoTarget;
		result.has_target = false;

		return result;
	}

} // namespace app::auto_aim
