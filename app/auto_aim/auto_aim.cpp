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
		 *
		 * @param armor 已由 Solver 成功填充 3D 字段的装甲板。
		 * @param timestamp_s 来自 FrameContext 的时间戳。
		 * @param source_detection_index 原始 detection 下标（Detector NMS 后顺序）。
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
		 * @brief pre-tracker 确定性候选排序（临时策略）。
		 *
		 * 仅对成功 PnP 集合排序，用于选出第一个兼容 target。
		 * 不再控制 PnP 的调用顺序：PnP 现在按 Detector 原始顺序全量求解，
		 * 排序结果只决定从 solved 集合中选择哪一个作为 AimResult::target。
		 *
		 * 排序关键字：
		 *   第一关键字：ArmorPriority，数字越小优先级越高；
		 *   第二关键字：距离图像中心越近越优先。
		 *
		 * 已知限制（重要）：当前 YOLO Detector 输出的 Armor.priority
		 * 全部为 ArmorPriority::Unknown，尚未实现 priority 来源。
		 * 因此当前实际 fallback 等价于 image-center ordering。
		 * 本阶段不得自行发明 priority mapping；
		 * sp_vision_25 的 priority 来源在后续 Tracker Plan 中单独追踪。
		 *
		 * 注意：这是 pre-tracker 阶段的临时策略，
		 *       未来会由真正的 Tracker / target association 替换。
		 *
		 * @param armors 候选装甲板（Detector 已按敌人颜色/置信度/几何过滤）。
		 * @param image_center 当前输入图像中心，禁止硬编码。
		 * @return 已按 comparator 排好序的候选 index；armors 为空时返回空 vector。
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

	AimResult AutoAim::process(const FrameContext& frame, AutoAimDebugData* debug)
	{
		// 每帧先清空 debug，保证 NoFrame / Error / NoTarget 等 early return
		// 不会残留上一帧数据。debug 为旁路观察，不改变算法行为。
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

		// 旁路观察：拷贝 Detector 原始输出（Solver 修改之前）。
		// 仅 debug 模式下发生，正常运行（debug == nullptr）零开销。
		if(debug != nullptr)
		{
			debug->detected_armors = detection.armors;
			debug->inference_time_ms = detection.inference_time_ms;
			debug->postprocess_time_ms = detection.postprocess_time_ms;
		}

		if(detection.armors.empty())
		{
			result.state = AimState::NoTarget;
			LOG_DEBUG(kLogModule, "frame {}: no armor detected", frame_count_);
			return result;
		}

		// ---- 2. pre-tracker 确定性候选排序 ----
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

		// 记录每个原始 index 是否 PnP 成功。使用 uint8_t 而非 vector<bool>，
		// 避免 vector<bool> 的位域/引用特殊语义。
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

			// 单个 PnP 几何失败不视为全局错误，继续处理下一候选。
			solved[index] = 1U;

			// index 是 detection.armors（即 debug.detected_armors）的原始下标，
			// 不是 order_candidates() 排序后的 rank。
			result.observations.push_back(
			    make_observation(candidate, frame.timestamp_s, index));

			if(debug != nullptr)
			{
				debug->solved_armor_indices.push_back(index);
			}
		}

		// ---- 4. pre-tracker 兼容目标选择：只从 PnP 成功集合中选择 ----
		for(const std::size_t index: order)
		{
			if(solved[index] == 0U)
			{
				continue;
			}

			Armor& candidate = detection.armors[index];

			if(debug != nullptr)
			{
				debug->selected_armor_index = index;
			}

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