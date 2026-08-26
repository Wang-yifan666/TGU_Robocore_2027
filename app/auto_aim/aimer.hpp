/**
 * @file aimer.hpp
 * @brief Aimer：延迟预测 + 弹道解算 + 选板，输出云台 yaw/pitch。
 *
 * 迁移自 sp_vision_25/tasks/auto_aim/aimer.{hpp,cpp}，保留其算法语义：
 * - 延迟预测（signed yaw_rate 比较高低速 delay）；
 * - 未来装甲板预测（纯几何，复用 vehicle_prediction）；
 * - 装甲板选择（SP25 兼容选板 + armor lock 迟滞）；
 * - 无空气阻力弹道解算（tools::Trajectory）；
 * - 按飞行时间迭代求解命中时刻；
 * - 输出最终 yaw/pitch（含 yaw_offset/pitch_offset）。
 *
 * Aimer 不负责：Detector / PnP / 数据关联 / 串口收发 / io::Command / 开火。
 * 不读取系统时钟：t_now 由调用方显式传入，保证 deterministic。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_AIMER_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_AIMER_HPP

#include <cstdint>
#include <optional>

#include <Eigen/Dense>

#include "app/auto_aim/tracker_types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 非法弹速处理策略。
	 */
	enum class InvalidBulletSpeedPolicy : std::uint8_t
	{
		Fallback = 0, ///< 使用 fallback_bullet_speed_mps 继续计算（SP25 兼容）。
		FailSafe      ///< 返回 invalid aiming solution。
	};

	/**
	 * @brief 选板策略。
	 *
	 * 本阶段只实现 SpCompat；predictive_hysteresis 留作未来扩展。
	 */
	enum class ArmorSwitchStrategy : std::uint8_t
	{
		SpCompat = 0
	};

	/**
	 * @brief Aimer 运行参数。
	 *
	 * 所有角度单位 rad、时间单位 s、速度单位 m/s、角速度单位 rad/s、长度单位 m。
	 * 生产 TOML 由 load_aimer_config* 显式加载，所有字段 required；
	 * make_default_aimer_config() 仅用于 programmatic / 单元测试。
	 */
	struct AimerConfig
	{
		double yaw_offset_rad = 0.0;
		double pitch_offset_rad = 0.0;

		double coming_angle_rad = 0.0;
		double leaving_angle_rad = 0.0;

		double outpost_coming_angle_rad = 0.0;
		double outpost_leaving_angle_rad = 0.0;

		/// 非陀螺分支的"可射击范围"半角阈值（SP25 为 60° / 57.3）。
		double shootable_angle_threshold_rad = 0.0;

		double high_speed_delay_s = 0.0;
		double low_speed_delay_s = 0.0;

		/// 高低速决策阈值（rad/s），保持 SP25 signed comparison。
		double decision_speed_rad_s = 0.0;

		/// 非陀螺判断依据：true 用 abs(radius)，false 用 abs(yaw_rate)。
		bool use_radius_for_gyro_detection = true;
		double non_gyro_radius_threshold_m = 0.0;
		double non_gyro_yaw_rate_threshold_rad_s = 0.0;

		InvalidBulletSpeedPolicy invalid_bullet_speed_policy = InvalidBulletSpeedPolicy::Fallback;
		double min_valid_bullet_speed_mps = 0.0;
		double fallback_bullet_speed_mps = 0.0;

		/// refinement 迭代次数（不含 initial trajectory solve）。
		int max_refinement_iterations = 0;
		double flight_time_convergence_s = 0.0;

		ArmorSwitchStrategy armor_switch_strategy = ArmorSwitchStrategy::SpCompat;
	};

	/**
	 * @brief 构造 SP25 兼容默认配置（仅测试/演示用，非生产 silent fallback）。
	 */
	AimerConfig make_default_aimer_config();

	/**
	 * @brief 校验 AimerConfig；非法时抛 std::invalid_argument。
	 */
	void validate_aimer_config(const AimerConfig& config);

	/**
	 * @brief Aimer 输出状态。
	 */
	enum class AimStatus : std::uint8_t
	{
		Success = 0,
		InvalidTarget,      ///< 状态/时间/目标数据非法。
		InvalidBulletSpeed, ///< fail_safe 且弹速非法。
		BallisticUnsolvable,
		NoValidArmor,
		PredictionFailed ///< 前推产生非有限结果（非"未收敛"）。
	};

	/**
	 * @brief Aimer 生产输出（供 task / shooter 消费）。
	 */
	struct AimingSolution
	{
		bool valid = false;
		AimStatus status = AimStatus::InvalidTarget;

		double yaw_rad = 0.0;
		double pitch_rad = 0.0;

		std::optional<int> selected_armor_id;

		/// 本阶段恒为 false；动态 shooting window 实现后才可能为 true。
		bool fire_allowed = false;
	};

	/**
	 * @brief Aimer 诊断输出（旁路观察用，不影响算法行为）。
	 *
	 * 仅在调用方显式传入非空指针时填充，对齐 AutoAimDebugData 风格。
	 */
	struct AimerDebugData
	{
		double t_muzzle_s = 0.0;
		double t_hit_s = 0.0;
		double flight_time_s = 0.0;

		Eigen::Vector3d aim_point_in_world = Eigen::Vector3d::Zero();

		int refinement_iterations = 0;
		bool ballistic_converged = false;
	};

	/**
	 * @brief Aimer：从 TrackedTarget 计算云台瞄准解。
	 */
	class Aimer
	{
	public:
		explicit Aimer(const AimerConfig& config);

		/**
		 * @brief 计算瞄准解。
		 *
		 * @param target Tracker 输出快照（t_state = target.timestamp_s）。
		 * @param t_now_s 本轮计算时刻（s，由调用方显式传入）。
		 * @param bullet_speed_mps 弹丸初速度（m/s）。
		 * @param debug 可选诊断输出。
		 */
		AimingSolution aim(const TrackedTarget& target, double t_now_s, double bullet_speed_mps,
		                   AimerDebugData* debug = nullptr);

		/**
		 * @brief 清空 armor lock 状态与 target token 作用域。
		 */
		void reset();

	private:
		AimerConfig config_;

		std::optional<int> lock_id_;
		std::optional<std::uint64_t> active_target_token_;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_AIMER_HPP
