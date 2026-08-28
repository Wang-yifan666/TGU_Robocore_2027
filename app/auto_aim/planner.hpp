/**
 * @file planner.hpp
 * @brief Planner：基于 TinyMPC 的云台 yaw/pitch 轨迹规划。
 *
 * 职责（仅此而已）：
 * - 用 Aimer preview 构造 past→now→future 的 reference trajectory；
 * - yaw / pitch 独立双积分器 MPC；
 * - 输出 position / velocity / acceleration。
 *
 * 不负责：fire / IO / board protocol / 弹道 / 选板 / 弹速 fallback。
 * 不持有 Aimer 引用（Aimer 仅在 plan() 调用期传入）。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_PLANNER_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_PLANNER_HPP

#include <array>
#include <cstdint>

#include <Eigen/Core>

#include "app/auto_aim/aimer.hpp"
#include "app/auto_aim/tracker_types.hpp"
#include "tools/tiny_mpc_2d.hpp"

namespace app::auto_aim
{

	inline constexpr double kPlannerDt = 0.01;
	inline constexpr int kPlannerHalfHorizon = 50;
	inline constexpr int kPlannerHorizon = kPlannerHalfHorizon * 2; // 100

	/**
	 * @brief Planner MPC 配置。
	 *
	 * 只含 MPC 自身参数；弹速 / delay / offset / fire threshold 属于其他模块。
	 */
	struct PlannerConfig
	{
		double max_yaw_acceleration_rad_s2 = 50.0;
		double max_pitch_acceleration_rad_s2 = 100.0;

		Eigen::Vector2d q_yaw = Eigen::Vector2d(9e6, 0.0);
		double r_yaw = 1.0;

		Eigen::Vector2d q_pitch = Eigen::Vector2d(9e6, 0.0);
		double r_pitch = 1.0;

		double rho = 1.0;
		int max_iter = 10;
	};

	PlannerConfig make_default_planner_config();

	/**
	 * @brief 校验 PlannerConfig；非法时抛 std::invalid_argument。
	 */
	void validate_planner_config(const PlannerConfig& config);

	/**
	 * @brief Planner 输出状态。
	 */
	enum class PlanningStatus : std::uint8_t
	{
		Success = 0,
		InvalidTarget,             ///< target 状态/数据非法，或 seed.token 不匹配。
		InvalidBulletSpeed,        ///< preview seed 未携带有效 effective bullet speed。
		ReferenceGenerationFailed, ///< preview 采样无法生成 reference。
		SolverFailure,             ///< TinyMPC 输出非有限 / 维度非法。
		InvalidConfig,
	};

	/**
	 * @brief 单次规划诊断（solver 状态，不作为判定 valid 的单一依据）。
	 */
	struct PlanningDiagnostics
	{
		int yaw_solve_code = 0;
		int pitch_solve_code = 0;
		bool yaw_solved = false;
		bool pitch_solved = false;
		int yaw_iterations = 0;
		int pitch_iterations = 0;

		double yaw_input_primal_residual = 0.0;
		double pitch_input_primal_residual = 0.0;

		double yaw_primal_max_abs_acc = 0.0;
		double yaw_projected_max_abs_acc = 0.0;
		double pitch_primal_max_abs_acc = 0.0;
		double pitch_projected_max_abs_acc = 0.0;

		double yaw_max_primal_projected_delta = 0.0;
		double pitch_max_primal_projected_delta = 0.0;

		double yaw_center_primal_u = 0.0;
		double yaw_center_projected_u = 0.0;
		double pitch_center_primal_u = 0.0;
		double pitch_center_projected_u = 0.0;

		/// primal work->x center 与"用 projected u 从同一 x0 重积分"的 center 位置差。
		double delta_yaw_center = 0.0;
		double delta_pitch_center = 0.0;

		double max_yaw_acceleration = 0.0;
		double max_pitch_acceleration = 0.0;
	};

	/**
	 * @brief Planner 旁路调试输出（test/characterization 用，默认 nullptr 零额外开销）。
	 */
	struct PlannerDebugData
	{
		/// 102 个 yaw angle samples（unwrap 后，连续）。
		std::array<double, kPlannerHorizon + 2> reference_yaw_samples{};
		/// 102 个 yaw angle samples（raw，unwrap 前，用于 ±π 穿越表征）。
		std::array<double, kPlannerHorizon + 2> reference_yaw_raw_samples{};
		/// 102 个 pitch angle samples。
		std::array<double, kPlannerHorizon + 2> reference_pitch_samples{};
		/// 100 个 yaw reference velocity（中心差分）。
		std::array<double, kPlannerHorizon> reference_yaw_velocity{};
		/// 100 个 pitch reference velocity（中心差分）。
		std::array<double, kPlannerHorizon> reference_pitch_velocity{};
		/// 每个 sample 的 selected armor id（-1 表示无）。
		std::array<int, kPlannerHorizon + 2> selected_armor_id{};

		double reference_generation_us = 0.0;
		double yaw_mpc_us = 0.0;
		double pitch_mpc_us = 0.0;
	};

	/**
	 * @brief Planner 输出（valid + status 风格，无 fire / 无旧 control）。
	 */
	struct PlanningSolution
	{
		bool valid = false;
		PlanningStatus status = PlanningStatus::InvalidTarget;

		double target_yaw_rad = 0.0;
		double target_pitch_rad = 0.0;

		double yaw_rad = 0.0;
		double yaw_velocity_rad_s = 0.0;
		double yaw_acceleration_rad_s2 = 0.0;

		double pitch_rad = 0.0;
		double pitch_velocity_rad_s = 0.0;
		double pitch_acceleration_rad_s2 = 0.0;

		PlanningDiagnostics diagnostics;
	};

	/**
	 * @brief Planner：TinyMPC 云台轨迹规划器。
	 */
	class Planner
	{
	public:
		explicit Planner(const PlannerConfig& config);

		/**
		 * @brief 围绕 seed.reference_center 规划 yaw/pitch 轨迹。
		 * @param seed 由本帧 Aimer::aim() 成功派生的 preview seed。
		 * @param target 与 seed 同 token 的 TrackedTarget。
		 * @param aimer 采样 provider（仅调用期使用，不保存引用）。
		 */
		PlanningSolution plan(const PlannerPreviewSeed& seed, const TrackedTarget& target,
		                      const Aimer& aimer, PlannerDebugData* debug = nullptr);

		/// 无跨帧状态（每帧 cold start）；保留接口对称。
		void reset();

	private:
		PlannerConfig config_;
		tools::TinyMpc2d yaw_solver_;
		tools::TinyMpc2d pitch_solver_;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_PLANNER_HPP
