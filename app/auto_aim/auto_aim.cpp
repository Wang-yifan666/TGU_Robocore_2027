#include "app/auto_aim/auto_aim.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
		 * @brief 从已完成 PnP 解算的 Armor 构造 Tracker 观测。
		 *
		 * 纯转换 helper：只负责复制 Solver 已完成的数据；不调 Solver、
		 * 不做 association / filtering、不获取系统时间、不修改 Armor。
		 */
		ArmorObservation make_observation(const Armor& armor, double timestamp_s,
		                                  std::size_t source_detection_index)
		{
			ArmorObservation observation;

			observation.color = armor.color;
			observation.name = armor.name;
			observation.type = armor.type;
			observation.priority = armor.priority;

			observation.confidence = armor.confidence;

			observation.position_in_gimbal = armor.xyz_in_gimbal;
			observation.position_in_world = armor.xyz_in_world;

			observation.ypd_in_world = armor.ypd_in_world;
			observation.armor_yaw_in_world = armor.ypr_in_world.x();

			observation.timestamp_s = timestamp_s;
			observation.source_detection_index = source_detection_index;

			return observation;
		}

		/**
		 * @brief legacy 确定性候选排序（只用于兼容 visible-armor 输出）。
		 *
		 * 注意：这是 pre-tracker 阶段的兼容策略，不再控制 Tracker association；
		 * Tracker 永远消费 AimResult::observations 全集。
		 */
		std::vector<std::size_t> order_candidates(const std::vector<Armor>& armors,
		                                          const cv::Point2f& image_center)
		{
			std::vector<std::size_t> order(armors.size());
			std::iota(order.begin(), order.end(), std::size_t{0});

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

			std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
				const Armor& lhs_armor = armors[lhs];
				const Armor& rhs_armor = armors[rhs];

				const int lhs_priority = priority_value(lhs_armor.priority);
				const int rhs_priority = priority_value(rhs_armor.priority);

				if(lhs_priority != rhs_priority)
				{
					return lhs_priority < rhs_priority;
				}

				return center_distance(lhs_armor) < center_distance(rhs_armor);
			});

			return order;
		}

	} // namespace

	AutoAim::AutoAim(Detector detector, Solver solver, Tracker tracker, Aimer aimer,
	                 Shooter shooter):
	detector_(std::move(detector)), solver_(std::move(solver)), tracker_(std::move(tracker)),
	aimer_(std::move(aimer)), shooter_(std::move(shooter))
	{
	}

	void AutoAim::reset()
	{
		frame_count_ = 0;
		tracker_.reset();
		aimer_.reset();
		shooter_.reset();
		last_target_token_.reset();
		LOG_INFO(kLogModule, "auto aim reset");
	}

	bool AutoAim::is_ready() const noexcept
	{
		return detector_.is_ready() && solver_.is_valid();
	}

	AimResult AutoAim::process(const FrameContext& frame, AutoAimDebugData* debug)
	{
		if(debug != nullptr)
		{
			*debug = AutoAimDebugData{};
		}

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

		if(debug != nullptr)
		{
			debug->detected_armors = detection.armors;
			debug->inference_time_ms = detection.inference_time_ms;
			debug->postprocess_time_ms = detection.postprocess_time_ms;
		}

		// ---- 2. legacy pre-tracker 候选排序（仅兼容输出）----
		const cv::Point2f image_center{frame.image.cols * 0.5F, frame.image.rows * 0.5F};

		const std::vector<std::size_t> order =
		    order_candidates(detection.armors, image_center);

		// ---- 3. 对所有 detection 按 Detector 原始顺序尝试 PnP ----
		solver_.set_r_gimbal_to_world(frame.q_imu_body_to_world);

		result.observations.reserve(detection.armors.size());

		if(debug != nullptr)
		{
			debug->solved_armor_indices.reserve(detection.armors.size());
		}

		std::vector<std::uint8_t> solved(detection.armors.size(), 0U);

		for(std::size_t index = 0; index < detection.armors.size(); ++index)
		{
			Armor& candidate = detection.armors[index];

			if(!solver_.solve(candidate))
			{
				LOG_DEBUG(kLogModule, "frame {}: PnP failed for candidate {}",
				          frame_count_, index);
				continue;
			}

			solved[index] = 1U;

			result.observations.push_back(
			    make_observation(candidate, frame.timestamp_s, index));

			if(debug != nullptr)
			{
				debug->solved_armor_indices.push_back(index);
			}
		}

		// ---- 4. legacy visible-armor 兼容输出（不控制 Tracker）----
		for(const std::size_t index: order)
		{
			if(solved[index] == 0U)
			{
				continue;
			}

			const Armor& candidate = detection.armors[index];

			if(debug != nullptr)
			{
				debug->selected_armor_index = index;
			}

			result.target = candidate;
			result.has_visible_target = true;

			const double distance = candidate.xyz_in_gimbal.norm();
			result.distance = distance;

			break;
		}

		// ---- 5. Tracker 消费所有 observations（空 observations 也驱动 miss）----
		const TrackResult track_result =
		    tracker_.track(result.observations, frame.timestamp_s);

		result.outcome = track_result.outcome;

		if(track_result.target)
		{
			const TrackedTarget& tracked = *track_result.target;

			result.tracked_target = tracked;

			switch(tracked.state)
			{
			case TrackerState::Tracking:
			case TrackerState::TempLost:
				result.state = AimState::Tracking;
				result.has_target = true;
				break;

			case TrackerState::Detecting:
				result.state = AimState::Detecting;
				result.has_target = false;
				break;

			case TrackerState::Lost:
			default:
				result.state = AimState::NoTarget;
				result.has_target = false;
				break;
			}
		}
		else
		{
			result.state = AimState::NoTarget;
			result.has_target = false;
		}

		// ---- 6. 目标身份作用域与射击历史 ----
		if(!result.has_target)
		{
			// Lost / Detecting：无确认目标，清空射击历史与目标身份。
			shooter_.reset();
			last_target_token_.reset();
		}
		else if(result.tracked_target.has_value())
		{
			const TrackedTarget& tracked = *result.tracked_target;

			if(last_target_token_ != tracked.target_token)
			{
				shooter_.reset();
				last_target_token_ = tracked.target_token;
			}

			// ---- 7. Aimer：Tracking / TempLost 都可产生瞄准解 ----
			const AimingSolution aiming =
			    aimer_.aim(tracked, frame.timestamp_s, frame.bullet_speed_mps);

			if(aiming.valid)
			{
				result.has_aim = true;
				result.yaw = aiming.yaw_rad;
				result.pitch = aiming.pitch_rad;

				// ---- 8. Shooter：仅 Tracking 允许开火 ----
				if(tracked.state == TrackerState::Tracking)
				{
					const double target_distance_m =
					    std::hypot(tracked.center_in_world.x(), tracked.center_in_world.y());
					result.fire = shooter_.shoot(aiming, target_distance_m,
					                              frame.gimbal_yaw_rad);
				}
				else
				{
					// TempLost：有 aim 但禁止开火；清空射击历史，回 Tracking 时重新建立。
					shooter_.reset();
					result.fire = false;
				}
			}
			else
			{
				// Aimer invalid：本帧无 aim、禁止开火，并 reset Shooter，
				// 使恢复后的第一帧重新建立历史并禁止开火。
				shooter_.reset();
				result.fire = false;
			}
		}

		return result;
	}

} // namespace app::auto_aim