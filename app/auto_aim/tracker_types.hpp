/**
 * @file tracker_types.hpp
 * @brief Detector/Solver 与未来 Tracker 之间的稳定数据边界。
 *
 * 本头文件定义 ArmorObservation：一块已完成三维解算的装甲板观测。
 * Tracker 后续只消费该类型，不依赖 cv::Mat、bbox、keypoints 或 Solver。
 *
 * 刻意不包含：
 * - bbox / four corners / cv::Mat pattern（2D 检测实现细节）
 * - Solver / Inference 依赖
 * - EKF / Target / Tracker 状态机（本阶段不实现）
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_TYPES_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_TYPES_HPP

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 已完成三维解算的单个装甲板观测（Tracker 的稳定输入边界）。
	 *
	 * 该类型只承载 Tracker 真正需要的信息：
	 * - identity / classification
	 * - confidence
	 * - 3D measurement（position）
	 * - orientation measurement
	 * - timestamp（来自 FrameContext，不由 Tracker 重新获取系统时间）
	 * - debug / source identity（可追溯到原始 detection index）
	 */
	struct ArmorObservation
	{
		ArmorColor color = ArmorColor::Unknown;
		ArmorName name = ArmorName::NotArmor;
		ArmorType type = ArmorType::Unknown;
		ArmorPriority priority = ArmorPriority::Unknown;

		double confidence = 0.0;

		/**
		 * @brief 装甲板在云台坐标系下的位置（单位 m）。
		 */
		Eigen::Vector3d position_in_gimbal = Eigen::Vector3d::Zero();

		/**
		 * @brief 装甲板在 "world" 坐标系下的位置（单位 m）。
		 *
		 * 坐标约定（当前 Solver）：
		 *   position_in_world = R_gimbal_to_world * position_in_gimbal
		 *
		 * 即：姿态稳定坐标。目前没有应用 gimbal / platform 在全局
		 * world frame 中的平移（global translation 不在本阶段引入）。
		 * 该约定仅作文档供未来 EKF 参考，不修改 Solver 坐标变换。
		 */
		Eigen::Vector3d position_in_world = Eigen::Vector3d::Zero();

		/**
		 * @brief world 系下 yaw / pitch / distance（球坐标，单位 rad / m）。
		 */
		Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();

		/**
		 * @brief world 系下的装甲板 yaw（单位 rad）。
		 *
		 * 来源：Armor::ypr_in_world.x()。
		 * 不绑定 Armor::yaw_raw（内部/历史语义，将来可能用于区分 raw / optimized yaw）。
		 */
		double armor_yaw_in_world = 0.0;

		double timestamp_s = 0.0;

		/**
		 * @brief 原始 detection 下标（DetectionResult::armors 的 NMS 后顺序）。
		 *
		 * 使用 std::numeric_limits<std::size_t>::max() 作为无效 sentinel，
		 * 避免把合法下标 0 误当作"未设置"。
		 */
		std::size_t source_detection_index = std::numeric_limits<std::size_t>::max();
	};

	/**
	 * @brief 由 Target 车辆模型生成的一块预测装甲板假设。
	 *
	 * 纯几何输出，不含 bbox / keypoints / cv::Mat。
	 * armor_id 表示该装甲板在车辆上的固定编号（0..armor_count-1）。
	 */
	struct ArmorHypothesis
	{
		int armor_id = 0;
		Eigen::Vector3d position_in_world = Eigen::Vector3d::Zero();
		double yaw_in_world = 0.0;
	};

	/**
	 * @brief Tracker 生命周期状态。
	 */
	enum class TrackerState : std::uint8_t
	{
		Lost = 0,
		Detecting,
		Tracking,
		TempLost
	};

	/**
	 * @brief Tracker 输出的稳定跟踪结果（车辆级，非可见装甲板）。
	 */
	struct TrackedTarget
	{
		TrackerState state = TrackerState::Lost;

		/// 本帧是否有 successful measurement（correction 成功）。
		/// false 时 innovation / nis / matched_* 均为 nullopt，
		/// 即使 EKF 内部存在 stale last_nis 也不输出。
		bool has_measurement = false;

		double timestamp_s = 0.0;

		ArmorColor color = ArmorColor::Unknown;
		ArmorName name = ArmorName::NotArmor;
		ArmorType type = ArmorType::Unknown;
		ArmorPriority priority = ArmorPriority::Unknown;

		Eigen::Vector3d center_in_world = Eigen::Vector3d::Zero();
		Eigen::Vector3d velocity_in_world = Eigen::Vector3d::Zero();

		double yaw = 0.0;
		double yaw_rate = 0.0;

		double radius = 0.0;
		double delta_radius = 0.0;
		double delta_z = 0.0;

		std::vector<ArmorHypothesis> predicted_armors;

		std::optional<std::size_t> matched_observation_index;
		std::optional<int> matched_armor_id;

		std::optional<Eigen::VectorXd> innovation;
		std::optional<double> nis;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_TRACKER_TYPES_HPP