/**
 * @file target.hpp
 * @brief 敌方车辆的 Target 数学模型（11D 车辆中心状态）。
 *
 * 本类描述整辆敌方车辆的 rotation center，而非当前可见的装甲板位置。
 * 仅依赖 Eigen 与 Tracker 边界类型（ArmorObservation / ArmorHypothesis），
 * 不依赖 cv::Mat / bbox / keypoints / Detector / Solver / OpenVINO。
 *
 * 状态布局（11 维）：
 *   0  center_x
 *   1  velocity_x
 *   2  center_y
 *   3  velocity_y
 *   4  center_z
 *   5  velocity_z
 *   6  yaw
 *   7  yaw_rate
 *   8  radius
 *   9  delta_radius
 *   10 delta_z
 *
 * 测量模型（4 维）：
 *   z = [world_x, world_y, world_z, armor_yaw_in_world]
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_TARGET_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_TARGET_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "app/auto_aim/tracker_types.hpp"
#include "app/auto_aim/types.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 状态向量维度。
	 */
	inline constexpr int kTargetStateDim = 11;

	/**
	 * @brief 测量向量维度。
	 */
	inline constexpr int kTargetMeasurementDim = 4;

	/**
	 * @brief 状态分量索引（与 kTargetStateDim 布局一致）。
	 */
	inline constexpr int kStateX = 0;
	inline constexpr int kStateVx = 1;
	inline constexpr int kStateY = 2;
	inline constexpr int kStateVy = 3;
	inline constexpr int kStateZ = 4;
	inline constexpr int kStateVz = 5;
	inline constexpr int kStateYaw = 6;
	inline constexpr int kStateYawRate = 7;
	inline constexpr int kStateRadius = 8;
	inline constexpr int kStateDeltaRadius = 9;
	inline constexpr int kStateDeltaZ = 10;

	/**
	 * @brief Target 过程噪声配置（全部来自 config，不散落 magic value）。
	 */
	struct TargetModelConfig
	{
		/// 平动加速度离散白噪声方差（作用在 x/y/z 三个平动自由度上）。
		double translation_accel_variance = 0.0;

		/// 偏航角加速度离散白噪声方差（作用在 yaw / yaw_rate 自由度上）。
		double yaw_accel_variance = 0.0;

		/// radius 随机游走方差（per-second variance）。
		double radius_random_walk_variance = 0.0;

		/// delta_radius 随机游走方差（per-second variance）。
		double delta_radius_random_walk_variance = 0.0;

		/// delta_z 随机游走方差（per-second variance）。
		double delta_z_random_walk_variance = 0.0;
	};

	/**
	 * @brief 由 identity 推导每辆车可见装甲板数量（deterministic 单测契约）。
	 *
	 * v1 规则：
	 *   Big && {Three, Four, Five} -> 2
	 *   Outpost                     -> 3
	 *   Base                        -> 3
	 *   else                        -> 4
	 */
	int armor_count_for(ArmorType type, ArmorName name);

	/**
	 * @brief 角度归一化到 [-pi, pi)。
	 */
	double wrap_angle(double angle);

	/**
	 * @brief 敌方车辆的 11D 车辆模型（rotation center）。
	 */
	class Target
	{
	public:
		/**
		 * @brief 目标模型服务函数：compute F (11x11) from dt。
		 */
		static Eigen::MatrixXd transition_matrix(double dt);

		/**
		 * @brief 目标模型服务函数：compute Q (11x11) from dt and config。
		 */
		static Eigen::MatrixXd process_noise_matrix(double dt, const TargetModelConfig& config);

		/**
		 * @brief 从单块装甲板观测初始化车辆 Target。
		 *
		 * @param observation 完成三维解算的装甲板观测（提供 identity / position / yaw）。
		 * @param initial_radius 初始车辆半径（由 config/profile 明确提供）。
		 * @param initial_covariance 11x11 初始协方差（由 config 明确提供）。
		 * @param config 过程噪声配置。
		 *
		 * @throw std::invalid_argument 观测不合法、初始半径非法、协方差 shape 非法。
		 */
		Target(const ArmorObservation& observation, double initial_radius,
		       const Eigen::MatrixXd& initial_covariance, const TargetModelConfig& config);

		/**
		 * @brief 恒定平动速度 + 恒定角速度的一阶预测（advance dt 秒）。
		 *
		 * @param dt 时间步长（秒），必须 finite 且 >= 0。
		 *
		 * @throw std::invalid_argument dt = NaN / Inf / < 0。
		 * @throw std::runtime_error 预测产生非有限结果（内部数值失败）。
		 */
		void predict(double dt);

		/**
		 * @brief 生成当前状态下的所有 armor hypotheses。
		 *
		 * armor_id 从 0..armor_count-1 递增，几何顺序固定。
		 */
		std::vector<ArmorHypothesis> armor_hypotheses() const;

		/**
		 * @brief 测量预测 h(x, armor_id) -> [x, y, z, yaw]（4 维）。
		 *
		 * 其中 armor_yaw 经 wrap_angle 归一化。
		 */
		Eigen::Vector4d measurement_model(const Eigen::VectorXd& x, int armor_id) const;

		/**
		 * @brief 解析 Jacobian dh/dx (4x11)。
		 */
		Eigen::MatrixXd measurement_jacobian(const Eigen::VectorXd& x, int armor_id) const;

		/**
		 * @brief 使用给定 armor_id 对观测做测量更新。
		 *
		 * 测量：[x, y, z, yaw]，使用 measurement_model / measurement_jacobian。
		 * yaw residual 通过 EKF residual hook wrap。
		 *
		 * @param observation 完成三维解算的装甲板观测。
		 * @param armor_id 匹配的装甲板编号。
		 * @param measurement_covariance 4x4 测量协方差（由 config 明确提供）。
		 *
		 * @return true 更新成功；false 数值失败（state/covariance 保持 prior）。
		 * @throw std::invalid_argument 输入非法 / 协方差 shape 非法 / armor_id 越界。
		 */
		bool correct(const ArmorObservation& observation, int armor_id,
		             const Eigen::MatrixXd& measurement_covariance);

		/**
		 * @brief 最近一次 successful correction 的 prior innovation（4 维）。
		 *
		 * 无 successful correction 时为空 vector。
		 */
		const Eigen::VectorXd& last_innovation() const noexcept;

		/**
		 * @brief 最近一次 successful correction 的 prior NIS。
		 *
		 * 无 successful correction 时为 NaN。
		 */
		double last_nis() const noexcept;

		/**
		 * @brief 当前后验状态（11 维）。
		 */
		const Eigen::VectorXd& state() const noexcept;

		/**
		 * @brief 当前后验协方差（11x11）。
		 */
		const Eigen::MatrixXd& covariance() const noexcept;

		int armor_count() const noexcept;

		ArmorColor color() const noexcept;
		ArmorName name() const noexcept;
		ArmorType type() const noexcept;
		ArmorPriority priority() const noexcept;

	private:
		/**
		 * @brief 给定 armor_id 计算 theta / r / use_alternate 等几何量。
		 */
		struct ArmorGeometry
		{
			double theta = 0.0;
			double radius = 0.0;
			bool use_alternate = false;
		};

		ArmorGeometry geometry(const Eigen::VectorXd& x, int armor_id) const;

		std::optional<tools::ExtendedKalmanFilter> ekf_;
		TargetModelConfig config_;

		ArmorColor color_ = ArmorColor::Unknown;
		ArmorName name_ = ArmorName::NotArmor;
		ArmorType type_ = ArmorType::Unknown;
		ArmorPriority priority_ = ArmorPriority::Unknown;
		int armor_count_ = 4;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_TARGET_HPP