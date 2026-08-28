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
	 * SpCompat 复现 SP25 兼容选板（含 armor lock 迟滞）；
	 * PredictiveHysteresis 使用提前观察 + 迟滞的离散切板。
	 */
	enum class ArmorSwitchStrategy : std::uint8_t
	{
		SpCompat = 0,
		PredictiveHysteresis
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

		/// PredictiveHysteresis 迟滞角（rad）。新板需比旧板至少优这么多才切换。
		double predictive_switch_hysteresis_rad = 0.0;

		/// PredictiveHysteresis 最大提前切板时间（s）。
		double predictive_switch_max_advance_s = 0.0;
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
	};

	/**
	 * @brief Aimer 诊断输出（旁路观察用，不影响算法行为）。
	 *
	 * 仅在调用方显式传入非空指针时填充，对齐 AutoAimDebugData 风格。
	 */
	struct AimerDebugData
	{
		double t_muzzle_s = 0.0;

		/// 最终 aim_point 对应的目标预测绝对时刻（= t_muzzle + 最终 prev_fly_time）。
		double target_prediction_time_s = 0.0;

		/// 弹丸到达时刻（= t_muzzle + 最终 current_traj.fly_time）。
		double ballistic_arrival_time_s = 0.0;

		/// 选板时刻（= target_prediction_time_s + switch_advance_s）。
		double armor_selection_time_s = 0.0;

		/// 提前切板时间（SpCompat 恒 0）。
		double switch_advance_s = 0.0;

		double flight_time_s = 0.0;

		Eigen::Vector3d aim_point_in_world = Eigen::Vector3d::Zero();

		int refinement_iterations = 0;
		bool ballistic_converged = false;

		/// 本帧调用前已提交的 predictive 状态（验证 transaction 用）。
		std::optional<int> previous_predictive_armor_id;

		/// 本帧最终 pending（提交后）的 predictive 状态。
		std::optional<int> pending_predictive_armor_id;
	};

	/**
	 * @brief Planner preview 使用的可演化局部状态快照（拷贝自 Aimer 跨帧状态）。
	 *
	 * 由 Planner 持有并逐 sample 演化；绝不影响 Aimer 成员。
	 */
	struct AimerPreviewState
	{
		std::optional<int> lock_id;                      ///< SpCompat armor lock。
		std::optional<int> predictive_selected_armor_id; ///< PredictiveHysteresis 上一帧提交值。
	};

	/**
	 * @brief 单次 preview 采样结果（在绝对时刻 t_s 的瞄准角）。
	 */
	struct AimSample
	{
		bool valid = false;
		AimStatus status = AimStatus::InvalidTarget;
		double yaw_rad = 0.0;    ///< = atan2(aim_point) + yaw_offset_rad。
		double pitch_rad = 0.0;  ///< = -(pitch + pitch_offset_rad)。
		Eigen::Vector3d aim_point = Eigen::Vector3d::Zero();
		std::optional<int> selected_armor_id;
	};

	/**
	 * @brief Aimer 成功求解后派生的 Planner preview seed。
	 *
	 * 由一次成功的 aim() 填充；携带 Planner 构造 reference 所需的全部时间/状态基准。
	 */
	struct PlannerPreviewSeed
	{
		std::uint64_t target_token = 0;           ///< 必须与 Planner 输入 target.target_token 一致。
		double effective_bullet_speed_mps = 0.0;  ///< Aimer 已 resolve 的实际弹速（fallback 后）。
		double reference_center_time_s = 0.0;     ///< 最终 target prediction time（含 delay + fly-time lead）。
		double reference_center_yaw_rad = 0.0;    ///< = 本帧 raw AimingSolution.yaw_rad（unwrap anchor）。
		double reference_center_pitch_rad = 0.0;  ///< = 本帧 raw AimingSolution.pitch_rad。
		AimerPreviewState preview_state;          ///< 本帧 aim transaction 修改选板状态之前的 baseline。
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
		 * @param seed 可选 planner preview seed（仅在 aim() 成功时填充）。
		 */
		AimingSolution aim(const TrackedTarget& target, double t_now_s, double bullet_speed_mps,
		                   AimerDebugData* debug = nullptr, PlannerPreviewSeed* seed = nullptr);

		/**
		 * @brief 无状态 preview 采样：在绝对时刻 t_s 求解瞄准角。
		 *
		 * 只演化调用方持有的 state（SpCompat 的 lock、PredictiveHysteresis 的 previous id），
		 * 绝不修改 Aimer 成员。供 Planner 逐 sample 构造 reference 使用。
		 */
		AimSample sample_at(AimerPreviewState& state, const TrackedTarget& target,
		                    double t_s, double bullet_speed_mps) const;

		/**
		 * @brief 清空 armor lock 状态与 target token 作用域。
		 */
		void reset();

	private:
		AimerConfig config_;

		std::optional<int> lock_id_;
		std::optional<int> predictive_selected_armor_id_;
		std::optional<std::uint64_t> active_target_token_;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_AIMER_HPP
