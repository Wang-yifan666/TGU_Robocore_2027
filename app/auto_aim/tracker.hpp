/**
 * @file tracker.hpp
 * @brief Tracker 生命周期状态机（Commit 5）。
 *
 * 输入：vector<ArmorObservation> + timestamp（来自 caller）。
 * 输出：TrackResult（outcome + optional<TrackedTarget>）。
 *
 * 状态机（detecting_confirm_hits == 1 时初始化帧直接进入 Tracking）：
 *   Lost ──valid observation──> Detecting ──enough hits──> Tracking
 *        <──too many misses─────┘                             │
 *                                                             │ miss
 *   Lost <──timeout──────────────────────────── TempLost <───┘
 *        └─reacquire────────────────────────────> Tracking
 *
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "app/auto_aim/association.hpp"
#include "app/auto_aim/target.hpp"
#include "app/auto_aim/tracker_types.hpp"
#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 按车辆类别映射初始半径（provisional migration defaults）。
	 *
	 * 这些值仅为 migration 初始值，必须在 replay 阶段验证，不作为真理写死。
	 */
	struct RadiusProfile
	{
		double balance_2 = 0.2;    ///< armor_count == 2（Big Three/Four/Five）
		double outpost_3 = 0.2765; ///< Outpost
		double base_3 = 0.3205;    ///< Base
		double default_4 = 0.2;    ///< 默认 4 装甲板
	};

	/**
	 * @brief Tracker 全部配置（显式提供，不含 TOML 读取）。
	 */
	struct TrackerConfig
	{
		/// Detecting 阶段确认进入 Tracking 所需的连续命中次数。
		int detecting_confirm_hits = 0;

		/// Detecting 阶段最大允许 miss 次数；超过则退回 Lost。
		int detecting_max_misses = 0;

		/// TempLost 阶段最大允许 miss 次数；超过则退回 Lost。
		int temp_lost_max_misses = 0;

		/// 允许正常预测的最大 dt（秒）。超过则视为旧 track 失效并 reset。
		double max_dt_s = 0.0;

		AssociationConfig association;

		/// 11x11 初始协方差（Target 构造用）。
		Eigen::MatrixXd initial_covariance;

		/// spherical 测量噪声配置（adaptive R）。
		MeasurementNoiseConfig measurement_noise;

		/// 过程噪声配置。
		TargetModelConfig process_noise;

		/// radius 合法域。
		double min_radius_m = 0.0;
		double max_radius_m = 0.0;

		RadiusProfile radius_profile;
	};

	/**
	 * @brief 校验 TrackerConfig 全部字段。
	 *
	 * 供 Tracker 构造与 tracker_config loader 复用；校验失败抛 std::invalid_argument。
	 */
	void validate_tracker_config(const TrackerConfig& config);

	/**
	 * @brief 构造一个可用于离线/演示/测试 composition 的 TrackerConfig。
	 *
	 * 注意：这是 test/demo fixture，不是 production TOML 装载；
	 * 生产路径应显式 load_tracker_config()。
	 */
	TrackerConfig make_default_tracker_config();

	/**
	 * @brief 单 Tracker 生命周期（LOST/DETECTING/TRACKING/TEMP_LOST）。
	 */
	class Tracker
	{
	public:
		explicit Tracker(const TrackerConfig& config);

		/**
		 * @brief 处理一帧观测。
		 *
		 * @param observations 本帧所有已完成 Solver 的装甲板观测（可为空）。
		 * @param timestamp_s 本帧时间戳（必须 finite，来自 caller，不读取系统时钟）。
		 *
		 * @return TrackResult（outcome 与 target 生命周期分离）。
		 *         outcome 始终反映本帧真实动作；Lost 时 target 为 nullopt。
		 *
		 * @throw std::invalid_argument timestamp 非 finite。
		 */
		TrackResult track(const std::vector<ArmorObservation>& observations, double timestamp_s);

		/**
		 * @brief 重置为 Lost / 清除 target。
		 */
		void reset();

	private:
		/**
		 * @brief 在 Lost 状态确定性选择一个 candidate（后续初始化可能进入
		 *        Detecting 或 Tracking，取决于 detecting_confirm_hits）。
		 *
		 * 排序：ArmorPriority（First 最优，Unknown 最后）→ world distance 小者
		 * → source_detection_index 小者 → observation vector index 小者。
		 */
		std::optional<ArmorObservation> select_initial_observation(
		    const std::vector<ArmorObservation>& observations) const;

		/**
		 * @brief 根据 identity 选择初始半径（依据 armor_count 规则 + RadiusProfile）。
		 */
		double radius_for(const ArmorObservation& observation) const;

		/**
		 * @brief 检查 radius / radius+delta_radius 是否落在配置合法域。
		 */
		bool geometry_healthy() const;

		/**
		 * @brief 从当前 target 构造输出快照（不含 measurement 字段）。
		 */
		TrackedTarget make_snapshot(TrackerState state, double timestamp_s) const;

		const TrackerConfig config_;

		TrackerState state_ = TrackerState::Lost;

		std::optional<Target> target_;

		std::uint64_t target_generation_ = 0;

		int hit_count_ = 0;
		int miss_count_ = 0;

		bool has_timestamp_ = false;
		double last_timestamp_s_ = 0.0;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_HPP