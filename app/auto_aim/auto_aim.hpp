/**
 * @file auto_aim.hpp
 * @brief 自瞄算法 facade / orchestrator（pre-tracker 阶段）。
 *
 * 本阶段职责：
 *   Detector -> Solver(all) -> ArmorObservation[] -> pre-tracker compatibility selection -> AimResult
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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "app/auto_aim/armor.hpp"
#include "app/auto_aim/detector/detector.hpp"
#include "app/auto_aim/solver.hpp"
#include "app/auto_aim/tracker.hpp"
#include "app/auto_aim/tracker_types.hpp"
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
		/**
		 * @brief 是否存在“可用”的车辆级 TrackedTarget。
		 *
		 * Commit 6 起语义升级为：
		 *   Tracking / TempLost -> true
		 *   Detecting / Lost    -> false
		 *
		 * 注意：false 不代表本帧没有可见装甲板；Detecting 阶段的
		 * 未确认 target 仍可通过 tracked_target 观察。
		 */
		bool has_target = false;

		AimState state = AimState::Idle;

		/**
		 * @brief legacy visible-armor compatibility output。
		 *
		 * 仅表达“当前可见 + PnP 成功的装甲板”的兼容信息，
		 * NOT vehicle tracking state。authoritative tracking result 见 tracked_target。
		 */
		Armor target;

		/**
		 * @brief 本帧所有成功完成 PnP 解算的装甲板观测。
		 *
		 * 顺序与 Detector 原始输出（NMS 后）一致，只包含 Solver 成功的
		 * detection；单个 PnP 失败不会阻止其它候选进入本列表。
		 * Tracker 后续只消费本列表，不依赖 target / Armor / cv::Mat。
		 */
		std::vector<ArmorObservation> observations;

		/**
		 * @brief 车辆级跟踪结果（Commit 6 起 authoritative tracking output）。
		 *
		 * 无 tracker 或 Lost 状态时为 nullopt。
		 */
		std::optional<TrackedTarget> tracked_target;

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
	 * @brief 单帧自瞄中间状态（旁路观察用，不改变算法行为）。
	 *
	 * 仅在调用方显式传入非空指针时填充；正常运行传入 nullptr 时零额外开销。
	 * 每帧 process() 开头都会清空 debug，保证 NoFrame / Error / NoTarget
	 * 等 early return 也不会残留上一帧数据。
	 */
	struct AutoAimDebugData
	{
		/**
		 * @brief Detector 原始输出（NMS 后、Solver 修改之前），
		 *        顺序与 DetectionResult::armors 完全一致。
		 *
		 * 注意：这些 Armor 的空间坐标（xyz_in_gimbal / xyz_in_world 等）
		 * 尚未被 Solver 填充。最终目标的空间信息请从 AimResult::target 获取。
		 */
		std::vector<Armor> detected_armors;

		/**
		 * @brief 最终被选中且 PnP 成功的装甲板在 detected_armors 中的下标。
		 *
		 * 语义：该下标是 Detector 输出（NMS 后）的原始顺序 index，
		 *       不是 order_candidates() 排序后的 rank。
		 * 无成功目标（NoTarget / NoFrame / Error）时保持空。
		 */
		std::optional<std::size_t> selected_armor_index;

		/**
		 * @brief 本帧所有 PnP 成功的 detection 下标（Detector 原始 index）。
		 *
		 * 语义：
		 * - 只记录 Solver 成功的 detection；
		 * - 顺序与 AimResult::observations 一致，且逐项满足
		 *   solved_armor_indices[i] == observations[i].source_detection_index；
		 * - 每帧通过 *debug = AutoAimDebugData{} 自动清空。
		 *
		 * 与 selected_armor_index 不同：solved_armor_indices 是成功集合，
		 * selected_armor_index 是 pre-tracker 最终选择。
		 */
		std::vector<std::size_t> solved_armor_indices;

		double inference_time_ms = 0.0;
		double postprocess_time_ms = 0.0;
	};

	/**
	 * @brief 自瞄 facade：持有 Detector 与 Solver，逐帧编排检测/选择/解算。
	 */
	class AutoAim
	{
	public:
		/**
		 * @brief 构造 AutoAim facade：持有 Detector / Solver / Tracker。
		 *
		 * AutoAim 不读取 TOML、不构造模型/硬件依赖；Tracker 已由
		 * task/composition 层根据 config 构造后注入。
		 */
		AutoAim(Detector detector, Solver solver, Tracker tracker);

		/**
		 * @brief 处理单帧输入。
		 * @param frame 单帧输入上下文（图像、时间戳、IMU 四元数）。
		 * @param debug 可选的旁路调试输出；为 nullptr 时正常实车路径零额外开销。
		 * @return 单帧自瞄结果。
		 */
		AimResult process(const FrameContext& frame, AutoAimDebugData* debug = nullptr);

		/**
		 * @brief 重置：frame_count 归零，并 reset Tracker 生命周期。
		 */
		void reset();

		bool is_ready() const noexcept;

	private:
		Detector detector_;
		Solver solver_;
		Tracker tracker_;

		std::uint64_t frame_count_ = 0;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_HPP