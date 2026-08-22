/**
 * @file association.hpp
 * @brief Target 与 ArmorObservation 的确定性关联（Commit 4 / v1）。
 *
 * 仅依赖 ArmorObservation / ArmorHypothesis / Target 预测信息，
 * 不依赖 Detector / Solver / cv::Mat / bbox / keypoints。
 *
 * v1 关联使用：
 *   identity hard gate + position hard gate + yaw hard gate + normalized residual score。
 * 不实现 NIS gating（posterior 不污染拒绝决策）。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_ASSOCIATION_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_ASSOCIATION_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "app/auto_aim/target.hpp"
#include "app/auto_aim/tracker_types.hpp"
#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 关联硬门限与打分尺度配置（全部显式提供）。
	 */
	struct AssociationConfig
	{
		/// 位置硬门限（m）。
		double max_position_error_m = 0.0;

		/// yaw 硬门限（rad）。
		double max_yaw_error_rad = 0.0;

		/// position score 尺度（m），必须 finite 且 > 0。
		double position_score_scale_m = 1.0;

		/// yaw score 尺度（rad），必须 finite 且 > 0。
		double yaw_score_scale_rad = 1.0;
	};

	/**
	 * @brief 一次成功关联的结果。
	 */
	struct AssociationResult
	{
		std::size_t observation_index = 0;
		int armor_id = 0;
		double score = 0.0;

		/// position residual = observation - hypothesis (world, 3D)。
		Eigen::Vector3d position_residual = Eigen::Vector3d::Zero();

		/// wrapped yaw residual。
		double yaw_residual = 0.0;
	};

	/**
	 * @brief 在 observations 中为 target 选择最优 (observation, armor_id) 关联。
	 *
	 * 规则：
	 * - identity（name / type / color）必须兼容；
	 * - position / yaw 通过硬门限；
	 * - 分数最低者胜出；tie-break：observation index 小者优先，再 armor_id 小者优先。
	 *
	 * @param target 车辆 target（用于 identity 与 armor hypotheses）。
	 * @param observations 本帧所有观测。
	 * @param config 关联配置。
	 *
	 * @return 无 valid pair 时为 std::nullopt。
	 */
	std::optional<AssociationResult> associate(const Target& target,
	                                           const std::vector<ArmorObservation>& observations,
	                                           const AssociationConfig& config);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_ASSOCIATION_HPP