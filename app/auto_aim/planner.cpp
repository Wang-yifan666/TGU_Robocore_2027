/**
 * @file planner.cpp
 * @brief Planner 实现：reference 生成 + yaw/pitch 独立 TinyMPC。
 */

#include "app/auto_aim/planner.hpp"

#include <array>
#include <cmath>
#include <string_view>

#include "tools/logger.hpp"
#include "tools/maths_tools.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "PLANNER";

		// 角度 sample 数量：HORIZON + 2（两侧各一个 ghost 用于中心差分）。
		constexpr int kSampleCount = kPlannerHorizon + 2;
		// reference center 在 angle samples 中的下标（= HALF_HORIZON + 1）。
		constexpr int kCenterSampleIndex = kPlannerHalfHorizon + 1;

		tools::TinyMpc2d::Config make_solver_config(const PlannerConfig& config, double max_acc,
		                                            const Eigen::Vector2d& q, double r)
		{
			tools::TinyMpc2d::Config solver_config;
			solver_config.dt = kPlannerDt;
			solver_config.rho = config.rho;
			solver_config.q = q;
			solver_config.r = r;
			solver_config.max_acceleration = max_acc;
			solver_config.horizon = kPlannerHorizon;
			solver_config.max_iter = config.max_iter;
			return solver_config;
		}

		double max_abs_projected(const tools::TinyMpc2d& solver)
		{
			double m = 0.0;

			for(int k = 0; k < solver.horizon() - 1; ++k)
			{
				m = std::max(m, std::abs(solver.acceleration(k)));
			}

			return m;
		}

		double max_abs_primal(const tools::TinyMpc2d& solver)
		{
			double m = 0.0;

			for(int k = 0; k < solver.horizon() - 1; ++k)
			{
				m = std::max(m, std::abs(solver.primal_acceleration(k)));
			}

			return m;
		}

	} // namespace

	Planner::Planner(const PlannerConfig& config):
	    config_(config),
	    yaw_solver_(make_solver_config(config, config.max_yaw_acceleration_rad_s2, config.q_yaw,
	                                   config.r_yaw)),
	    pitch_solver_(make_solver_config(config, config.max_pitch_acceleration_rad_s2, config.q_pitch,
	                                     config.r_pitch))
	{
		// 配置非法由 TinyMpc2d 构造时 fail-fast。
	}

	void Planner::reset()
	{
		// 无跨帧状态：每次 plan() 都从 reference 重新 seed x0（cold start）。
	}

	PlanningSolution Planner::plan(const PlannerPreviewSeed& seed, const TrackedTarget& target,
	                               const Aimer& aimer)
	{
		PlanningSolution solution;
		solution.status = PlanningStatus::InvalidTarget;
		solution.diagnostics.max_yaw_acceleration = config_.max_yaw_acceleration_rad_s2;
		solution.diagnostics.max_pitch_acceleration = config_.max_pitch_acceleration_rad_s2;

		// token 一致性（preview baseline 的 target scope 必须匹配）。
		if(seed.target_token != target.target_token)
		{
			LOG_DEBUG(kLogModule, "seed.target_token mismatch");
			return solution;
		}

		// effective bullet speed（Aimer 已 resolve；Planner 不再解释 raw bullet speed）。
		if(!std::isfinite(seed.effective_bullet_speed_mps) || seed.effective_bullet_speed_mps <= 0.0)
		{
			solution.status = PlanningStatus::InvalidBulletSpeed;
			return solution;
		}

		if(!std::isfinite(seed.reference_center_time_s))
		{
			solution.status = PlanningStatus::InvalidTarget;
			return solution;
		}

		// ---- 102 个 angle samples（past ghost → ... → center → ... → future ghost）----
		std::array<double, kSampleCount> raw_yaw{};
		std::array<double, kSampleCount> pitch{};

		AimerPreviewState state = seed.preview_state;

		for(int j = 0; j < kSampleCount; ++j)
		{
			const double t_j = seed.reference_center_time_s + (j - kCenterSampleIndex) * kPlannerDt;
			const AimSample sample =
			    aimer.sample_at(state, target, t_j, seed.effective_bullet_speed_mps);

			if(!sample.valid)
			{
				LOG_DEBUG(kLogModule, "reference sample {} invalid", j);
				solution.status = PlanningStatus::ReferenceGenerationFailed;
				return solution;
			}

			raw_yaw[j] = sample.yaw_rad;
			pitch[j] = sample.pitch_rad;
		}

		// ---- yaw unwrap（anchor = raw aiming yaw = seed.reference_center_yaw_rad）----
		const double anchor = seed.reference_center_yaw_rad;
		std::array<double, kSampleCount> unwrapped_yaw{};
		unwrapped_yaw[kCenterSampleIndex] = anchor;

		for(int j = kCenterSampleIndex + 1; j < kSampleCount; ++j)
		{
			unwrapped_yaw[j] = unwrapped_yaw[j - 1]
			    + tools::maths_tools::limit_rad(raw_yaw[j] - raw_yaw[j - 1]);
		}

		for(int j = kCenterSampleIndex - 1; j >= 0; --j)
		{
			unwrapped_yaw[j] = unwrapped_yaw[j + 1]
			    - tools::maths_tools::limit_rad(raw_yaw[j + 1] - raw_yaw[j]);
		}

		// ---- reference（yaw 相对 anchor；pitch 无需 unwrap）----
		Eigen::Matrix<double, 2, kPlannerHorizon> yaw_ref;
		Eigen::Matrix<double, 2, kPlannerHorizon> pitch_ref;

		for(int i = 0; i < kPlannerHorizon; ++i)
		{
			yaw_ref(0, i) = unwrapped_yaw[i + 1] - anchor;
			yaw_ref(1, i) = (unwrapped_yaw[i + 2] - unwrapped_yaw[i]) / (2.0 * kPlannerDt);
			pitch_ref(0, i) = pitch[i + 1];
			pitch_ref(1, i) = (pitch[i + 2] - pitch[i]) / (2.0 * kPlannerDt);
		}

		// ---- x0 = reference horizon 样本 0（非实测云台状态）----
		const Eigen::Vector2d yaw_x0(yaw_ref(0, 0), yaw_ref(1, 0));
		const Eigen::Vector2d pitch_x0(pitch_ref(0, 0), pitch_ref(1, 0));

		// ---- solve（cold start）----
		const int yaw_code = yaw_solver_.solve(yaw_x0, yaw_ref);
		const int pitch_code = pitch_solver_.solve(pitch_x0, pitch_ref);

		// ---- 输出 ----
		solution.target_yaw_rad =
		    tools::maths_tools::limit_rad(yaw_ref(0, kPlannerHalfHorizon) + anchor);
		solution.target_pitch_rad = pitch_ref(0, kPlannerHalfHorizon);

		solution.yaw_rad =
		    tools::maths_tools::limit_rad(yaw_solver_.position(kPlannerHalfHorizon) + anchor);
		solution.yaw_velocity_rad_s = yaw_solver_.velocity(kPlannerHalfHorizon);
		solution.yaw_acceleration_rad_s2 = yaw_solver_.acceleration(kPlannerHalfHorizon);

		solution.pitch_rad = pitch_solver_.position(kPlannerHalfHorizon);
		solution.pitch_velocity_rad_s = pitch_solver_.velocity(kPlannerHalfHorizon);
		solution.pitch_acceleration_rad_s2 = pitch_solver_.acceleration(kPlannerHalfHorizon);

		// ---- diagnostics ----
		solution.diagnostics.yaw_solve_code = yaw_code;
		solution.diagnostics.pitch_solve_code = pitch_code;
		solution.diagnostics.yaw_solved = yaw_solver_.solved();
		solution.diagnostics.pitch_solved = pitch_solver_.solved();
		solution.diagnostics.yaw_iterations = yaw_solver_.iteration_count();
		solution.diagnostics.pitch_iterations = pitch_solver_.iteration_count();
		solution.diagnostics.yaw_primal_max_abs_acc = max_abs_primal(yaw_solver_);
		solution.diagnostics.yaw_projected_max_abs_acc = max_abs_projected(yaw_solver_);
		solution.diagnostics.pitch_primal_max_abs_acc = max_abs_primal(pitch_solver_);
		solution.diagnostics.pitch_projected_max_abs_acc = max_abs_projected(pitch_solver_);

		// ---- acceptance：有限 + reference 有限；return code 仅 diagnostic ----
		const bool finite = std::isfinite(solution.yaw_rad)
		    && std::isfinite(solution.yaw_velocity_rad_s)
		    && std::isfinite(solution.yaw_acceleration_rad_s2) && std::isfinite(solution.pitch_rad)
		    && std::isfinite(solution.pitch_velocity_rad_s)
		    && std::isfinite(solution.pitch_acceleration_rad_s2) && yaw_ref.allFinite()
		    && pitch_ref.allFinite();

		if(!finite)
		{
			solution.status = PlanningStatus::SolverFailure;
			return solution;
		}

		solution.valid = true;
		solution.status = PlanningStatus::Success;
		return solution;
	}

} // namespace app::auto_aim
