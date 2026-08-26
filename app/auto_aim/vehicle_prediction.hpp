/**
 * @file vehicle_prediction.hpp
 * @brief 从 TrackedTarget 快照前推整车状态并生成装甲板假设（纯几何 helper）。
 *
 * 与 Target::predict(dt) + Target::armor_hypotheses() 的几何公式保持一致，
 * 供 Aimer 在任意未来时刻 t 生成装甲板位姿，而不依赖 EKF / Target 内部。
 *
 * 本文件刻意保持独立，避免为了去重而对 Target 做大范围重构；
 * 通过 equivalence test 防止与 Target 几何漂移。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_VEHICLE_PREDICTION_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_VEHICLE_PREDICTION_HPP

#include <vector>

#include <Eigen/Dense>

#include "app/auto_aim/target.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 前推后的整车几何状态（不含速度，只保留生成装甲板所需量）。
	 */
	struct PredictedVehicle
	{
		Eigen::Vector3d center = Eigen::Vector3d::Zero(); ///< 整车中心 (m)。
		double yaw = 0.0;                                ///< 整车偏航角 (rad)。
		double radius = 0.0;                             ///< 半径 (m)。
		double delta_radius = 0.0;                       ///< 交替半径增量 (m)。
		double delta_z = 0.0;                            ///< 交替高度增量 (m)。
		int armor_count = 0;                             ///< 装甲板数量。
	};

	/**
	 * @brief 从 TrackedTarget 快照前推 dt_s 秒后的整车几何状态。
	 *
	 * 模型：center += velocity * dt，yaw += yaw_rate * dt（yaw 归一化到 [-π, π)）。
	 * 前置条件：dt_s >= 0 且有限；调用方（Aimer）负责保证。
	 */
	PredictedVehicle predict_vehicle(const TrackedTarget& target, double dt_s);

	/**
	 * @brief 由 PredictedVehicle 生成全部装甲板假设（armor_id 0..count-1）。
	 *
	 * 几何与 Target::armor_hypotheses() 完全一致：
	 * theta = wrap(yaw + id * 2π / count)；
	 * 仅 count == 4 且 id ∈ {1,3} 时使用 radius + delta_radius 与 z + delta_z。
	 */
	std::vector<ArmorHypothesis> armor_hypotheses(const PredictedVehicle& vehicle);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_VEHICLE_PREDICTION_HPP
