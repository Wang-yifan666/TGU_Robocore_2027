#include "app/auto_aim/auto_aim.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "tools/logger.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "AUTO_AIM";

		/**
		 * @brief pre-tracker 确定性目标选择（临时策略）。
		 *
		 * 复现旧 sp_vision_25 Tracker 在 lost 状态初始化目标时的排序行为：
		 *   第一关键字：ArmorPriority，数字越小优先级越高；
		 *   第二关键字：距离图像中心越近越优先。
		 *
		 * 注意：这是 pre-tracker 阶段的临时策略，
		 *       未来会由真正的 Tracker / target association 替换。
		 *
		 * @param armors 候选装甲板（Detector 已按敌人颜色/置信度/几何过滤）。
		 * @param image_center 当前输入图像中心，禁止硬编码。
		 * @return 最优候选的迭代器；armors 为空时返回 end()。
		 */
		std::vector<Armor>::const_iterator select_pre_tracker_target(
		    const std::vector<Armor>& armors, const cv::Point2f& image_center)
		{
			const auto priority_value = [](ArmorPriority priority) -> int {
				switch(priority)
				{
				case ArmorPriority::First:
					return 1;
				case ArmorPriority::Second:
					return 2;
				case ArmorPriority::Third:
					return 3;
				case ArmorPriority::Fourth:
					return 4;
				case ArmorPriority::Fifth:
					return 5;
				case ArmorPriority::Unknown:
				default:
					return 255;
				}
			};

			const auto center_distance = [&image_center](const Armor& armor) {
				const cv::Point2f& center = armor.center;
				const double dx = static_cast<double>(center.x) - static_cast<double>(image_center.x);
				const double dy = static_cast<double>(center.y) - static_cast<double>(image_center.y);
				return std::sqrt(dx * dx + dy * dy);
			};

			return std::min_element(
			    armors.begin(), armors.end(), [&](const Armor& lhs, const Armor& rhs) {
				    const int lhs_priority = priority_value(lhs.priority);
				    const int rhs_priority = priority_value(rhs.priority);

				    if(lhs_priority != rhs_priority)
				    {
					    return lhs_priority < rhs_priority;
				    }

				    return center_distance(lhs) < center_distance(rhs);
			    });
		}

	} // namespace

	AutoAim::AutoAim(Detector detector, Solver solver):
	detector_(std::move(detector)), solver_(std::move(solver))
	{
	}

	void AutoAim::reset()
	{
		frame_count_ = 0;
		LOG_INFO(kLogModule, "auto aim reset");
	}

	bool AutoAim::is_ready() const noexcept
	{
		return detector_.is_ready() && solver_.is_valid();
	}

	AimResult AutoAim::process(const FrameContext& frame)
	{
		AimResult result;
		result.timestamp_s = frame.timestamp_s;

		if(!is_ready())
		{
			result.state = AimState::Error;
			LOG_ERROR(kLogModule, "auto aim dependencies are not ready");
			return result;
		}

		if(frame.image.empty())
		{
			result.state = AimState::NoFrame;
			LOG_WARN(kLogModule, "input frame is empty");
			return result;
		}

		++frame_count_;

		// ---- 1. 检测 ----
		DetectionResult detection = detector_.detect(frame.image, frame_count_);

		if(detection.armors.empty())
		{
			result.state = AimState::NoTarget;
			LOG_DEBUG(kLogModule, "frame {}: no armor detected", frame_count_);
			return result;
		}

		// ---- 2. pre-tracker 确定性目标选择 ----
		const cv::Point2f image_center{frame.image.cols * 0.5F, frame.image.rows * 0.5F};

		const auto selected =
		    select_pre_tracker_target(detection.armors, image_center);

		// ---- 3. 按候选顺序尝试 PnP ----
		// 首选由 select_pre_tracker_target 决定。
		std::vector<std::size_t> order(detection.armors.size());
		std::iota(order.begin(), order.end(), std::size_t{0});

		if(selected != detection.armors.end())
		{
			const std::size_t first_index =
			    static_cast<std::size_t>(selected - detection.armors.begin());

			std::swap(order[0], *std::find(order.begin(), order.end(), first_index));
		}

		solver_.set_r_gimbal_to_world(frame.q_imu_body_to_world);

		for(const std::size_t index: order)
		{
			Armor& candidate = detection.armors[index];

			if(!solver_.solve(candidate))
			{
				LOG_DEBUG(kLogModule, "frame {}: PnP failed for candidate {}",
				          frame_count_, index);
				continue;
			}

			// 单个 PnP 几何失败不视为全局错误，继续尝试下一候选。
			result.has_target = true;
			result.state = AimState::Detecting;
			result.target = candidate;

			const double distance = candidate.xyz_in_gimbal.norm();
			result.distance = distance;
			// yaw / pitch 保留默认 0：raw observation，非最终云台指令。

			LOG_DEBUG(kLogModule,
			          "frame {}: solved target, name={}, type={}, color={}, "
			          "xyz_in_gimbal=({:.3f}, {:.3f}, {:.3f}) m, distance={:.3f} m",
			          frame_count_, to_string(candidate.name), to_string(candidate.type),
			          to_string(candidate.color), candidate.xyz_in_gimbal.x(),
			          candidate.xyz_in_gimbal.y(), candidate.xyz_in_gimbal.z(), result.distance);

			return result;
		}

		// 所有候选 PnP 均失败。
		result.state = AimState::NoTarget;
		LOG_DEBUG(kLogModule, "frame {}: all candidates failed PnP", frame_count_);
		return result;
	}

} // namespace app::auto_aim