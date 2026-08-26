/**
 * @file aimer.cpp
 * @brief Aimer 实现：延迟预测 + 选板 + 弹道迭代。
 *
 * 迁移自 sp_vision_25/tasks/auto_aim/aimer.cpp，保留 SP25 算法语义。
 */

#include "app/auto_aim/aimer.hpp"

#include <cmath>
#include <string_view>

#include "app/auto_aim/vehicle_prediction.hpp"
#include "tools/logger.hpp"
#include "tools/maths_tools.hpp"
#include "tools/trajectory.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "AIMER";

		bool finite_vehicle_state(const TrackedTarget& target)
		{
			return target.center_in_world.allFinite() && target.velocity_in_world.allFinite()
			    && std::isfinite(target.yaw) && std::isfinite(target.yaw_rate)
			    && std::isfinite(target.radius) && std::isfinite(target.delta_radius)
			    && std::isfinite(target.delta_z);
		}

		bool finite_hypotheses(const std::vector<ArmorHypothesis>& hypotheses)
		{
			for(const auto& h: hypotheses)
			{
				if(!h.position_in_world.allFinite() || !std::isfinite(h.yaw_in_world))
				{
					return false;
				}
			}

			return true;
		}

		/**
		 * @brief SP25 兼容选板 + armor lock 迟滞。
		 *
		 * 返回选中的装甲板假设；无有效板时返回 nullopt。
		 * lock_id 按引用传入，实现跨帧迟滞（作用域由 Aimer 的 target_token 管理）。
		 */
		std::optional<ArmorHypothesis> select_armor(const AimerConfig& config,
		                                            std::optional<int>& lock_id,
		                                            const PredictedVehicle& vehicle,
		                                            const std::vector<ArmorHypothesis>& hypotheses,
		                                            double yaw_rate, bool has_armor_switch,
		                                            ArmorName name)
		{
			if(hypotheses.empty())
			{
				return std::nullopt;
			}

			// 未发生过跳变：只有当前装甲板位置已知（SP25 jumped == false）。
			if(!has_armor_switch)
			{
				return hypotheses[0];
			}

			const double center_yaw = std::atan2(vehicle.center.y(), vehicle.center.x());

			std::vector<double> delta_angle_list;
			delta_angle_list.reserve(hypotheses.size());

			for(const auto& h: hypotheses)
			{
				delta_angle_list.push_back(
				    tools::maths_tools::limit_rad(h.yaw_in_world - center_yaw));
			}

			const int armor_num = static_cast<int>(hypotheses.size());

			// 非陀螺判断：默认 abs(radius)（SP25 兼容），use_radius=false 时 abs(yaw_rate)。
			bool non_gyro;

			if(config.use_radius_for_gyro_detection)
			{
				non_gyro = std::abs(vehicle.radius) <= config.non_gyro_radius_threshold_m;
			}
			else
			{
				non_gyro = std::abs(yaw_rate) <= config.non_gyro_yaw_rate_threshold_rad_s;
			}

			// 不考虑小陀螺（且非前哨站）。
			if(non_gyro && name != ArmorName::Outpost)
			{
				std::vector<int> id_list;

				for(int i = 0; i < armor_num; i++)
				{
					if(std::abs(delta_angle_list[i]) > config.shootable_angle_threshold_rad)
					{
						continue;
					}

					id_list.push_back(i);
				}

				if(id_list.empty())
				{
					LOG_WARN(kLogModule, "empty shootable armor id list");
					return std::nullopt;
				}

				// 锁定模式：防止在两块都接近可射击边界的装甲板之间来回切换。
				if(id_list.size() > 1)
				{
					const int id0 = id_list[0];
					const int id1 = id_list[1];

					if(lock_id != id0 && lock_id != id1)
					{
						lock_id = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1]))
						              ? id0
						              : id1;
					}

					return hypotheses[*lock_id];
				}

				// 只有一个装甲板在可射击范围内：退出锁定模式。
				lock_id.reset();
				return hypotheses[id_list[0]];
			}

			// 小陀螺 / 前哨站分支。
			const double coming_angle = (name == ArmorName::Outpost) ? config.outpost_coming_angle_rad
			                                                         : config.coming_angle_rad;
			const double leaving_angle = (name == ArmorName::Outpost)
			                                 ? config.outpost_leaving_angle_rad
			                                 : config.leaving_angle_rad;

			for(int i = 0; i < armor_num; i++)
			{
				if(std::abs(delta_angle_list[i]) > coming_angle)
				{
					continue;
				}

				if(yaw_rate > 0 && delta_angle_list[i] < leaving_angle)
				{
					return hypotheses[i];
				}

				if(yaw_rate < 0 && delta_angle_list[i] > -leaving_angle)
				{
					return hypotheses[i];
				}
			}

			return std::nullopt;
		}

		/**
		 * @brief PredictiveHysteresis 提前切板时间。
		 *
		 * 使用 abs(yaw_rate)，正反转在"提前时间长度"上完全对称。
		 */
		double compute_switch_advance(const AimerConfig& config, double yaw_rate)
		{
			constexpr double kEpsilon = 1e-9;
			const double abs_w = std::abs(yaw_rate);

			if(abs_w <= kEpsilon)
			{
				return 0.0;
			}

			return std::min(config.predictive_switch_max_advance_s,
			                config.predictive_switch_hysteresis_rad / abs_w);
		}

		/**
		 * @brief PredictiveHysteresis 选板（纯函数，不修改跨帧状态）。
		 *
		 * score_i = abs(limit_rad(armors_at_selection[i].yaw - center_yaw_at_predict))。
		 * 新板需比旧板严格好一个 hysteresis 角才切换（严格 <）。
		 */
		int predictive_select_armor(const AimerConfig& config,
		                            const std::vector<ArmorHypothesis>& armors_at_selection,
		                            const Eigen::Vector3d& center_at_predict, bool has_armor_switch,
		                            const std::optional<int>& previous_id)
		{
			if(!has_armor_switch || armors_at_selection.empty())
			{
				return 0; // SP25 jumped == false 语义：恒选 armor 0。
			}

			const double center_yaw = std::atan2(center_at_predict.y(), center_at_predict.x());

			const int armor_num = static_cast<int>(armors_at_selection.size());
			std::vector<double> scores(static_cast<std::size_t>(armor_num));

			int best_id = 0;

			for(int i = 0; i < armor_num; ++i)
			{
				scores[static_cast<std::size_t>(i)] = std::abs(tools::maths_tools::limit_rad(
				    armors_at_selection[static_cast<std::size_t>(i)].yaw_in_world - center_yaw));

				if(i == 0
				   || scores[static_cast<std::size_t>(i)] < scores[static_cast<std::size_t>(best_id)])
				{
					best_id = i;
				}
			}

			// Hysteresis：严格 < 才切换（新板需明显优于旧板）。
			if(previous_id.has_value() && *previous_id >= 0 && *previous_id < armor_num
			   && best_id != *previous_id)
			{
				if(!(scores[static_cast<std::size_t>(best_id)]
				         + config.predictive_switch_hysteresis_rad
				     < scores[static_cast<std::size_t>(*previous_id)]))
				{
					return *previous_id;
				}
			}

			return best_id;
		}

		/**
		 * @brief 单次"在 t_predict 时刻求解"的结果。
		 */
		struct SolveResult
		{
			bool valid = false;
			AimStatus status = AimStatus::InvalidTarget;
			int selected_armor_id = 0;
			Eigen::Vector3d aim_point = Eigen::Vector3d::Zero();
			double fly_time = 0.0;
			double pitch = 0.0;
			double target_prediction_time_s = 0.0;
			double armor_selection_time_s = 0.0;
			double switch_advance_s = 0.0;
		};

		/**
		 * @brief 在 t_predict 时刻完成一次"预测 -> 选板 -> 弹道"求解。
		 *
		 * 选板与瞄准严格分离：选板可提前（t_selection），
		 * 但 aim_point 恒取自 armors_at_predict[selected_id]（不提前）。
		 */
		SolveResult solve_at_prediction_time(const AimerConfig& config, std::optional<int>& lock_id,
		                                     const std::optional<int>& previous_predictive_id,
		                                     const TrackedTarget& target, double bullet_speed,
		                                     double t_predict)
		{
			SolveResult result;
			result.target_prediction_time_s = t_predict;

			const double dt_predict = t_predict - target.timestamp_s;
			const PredictedVehicle vehicle_at_predict = predict_vehicle(target, dt_predict);

			if(!vehicle_at_predict.center.allFinite())
			{
				result.status = AimStatus::PredictionFailed;
				return result;
			}

			const std::vector<ArmorHypothesis> armors_at_predict =
			    armor_hypotheses(vehicle_at_predict);

			if(!finite_hypotheses(armors_at_predict))
			{
				result.status = AimStatus::PredictionFailed;
				return result;
			}

			int selected_id = 0;
			double switch_advance = 0.0;
			double t_selection = t_predict;

			if(config.armor_switch_strategy == ArmorSwitchStrategy::SpCompat)
			{
				const std::optional<ArmorHypothesis> aim =
				    select_armor(config, lock_id, vehicle_at_predict, armors_at_predict,
				                 target.yaw_rate, target.has_armor_switch, target.name);

				if(!aim)
				{
					result.status = AimStatus::NoValidArmor;
					return result;
				}

				selected_id = aim->armor_id;
			}
			else
			{
				switch_advance = compute_switch_advance(config, target.yaw_rate);
				t_selection = t_predict + switch_advance;

				const double dt_selection = t_selection - target.timestamp_s;
				const PredictedVehicle vehicle_at_selection = predict_vehicle(target, dt_selection);

				if(!vehicle_at_selection.center.allFinite())
				{
					result.status = AimStatus::PredictionFailed;
					return result;
				}

				const std::vector<ArmorHypothesis> armors_at_selection =
				    armor_hypotheses(vehicle_at_selection);

				if(!finite_hypotheses(armors_at_selection))
				{
					result.status = AimStatus::PredictionFailed;
					return result;
				}

				selected_id =
				    predictive_select_armor(config, armors_at_selection, vehicle_at_predict.center,
				                            target.has_armor_switch, previous_predictive_id);
			}

			if(selected_id < 0 || selected_id >= static_cast<int>(armors_at_predict.size()))
			{
				result.status = AimStatus::NoValidArmor;
				return result;
			}

			result.selected_armor_id = selected_id;
			result.armor_selection_time_s = t_selection;
			result.switch_advance_s = switch_advance;

			// 瞄准点恒来自 armors_at_predict（不提前）。
			result.aim_point =
			    armors_at_predict[static_cast<std::size_t>(selected_id)].position_in_world;

			const double d = std::hypot(result.aim_point.x(), result.aim_point.y());
			const tools::Trajectory trajectory(bullet_speed, d, result.aim_point.z());

			if(trajectory.unsolvable)
			{
				result.status = AimStatus::BallisticUnsolvable;
				return result;
			}

			result.fly_time = trajectory.fly_time;
			result.pitch = trajectory.pitch;
			result.valid = true;
			result.status = AimStatus::Success;
			return result;
		}

	} // namespace

	Aimer::Aimer(const AimerConfig& config): config_(config)
	{
		validate_aimer_config(config_);
	}

	void Aimer::reset()
	{
		lock_id_.reset();
		predictive_selected_armor_id_.reset();
		active_target_token_.reset();
	}

	AimingSolution Aimer::aim(const TrackedTarget& target, double t_now_s, double bullet_speed_mps,
	                          AimerDebugData* debug)
	{
		if(debug != nullptr)
		{
			*debug = AimerDebugData{};
		}

		AimingSolution solution;
		solution.status = AimStatus::InvalidTarget;

		// ---- 防御性检查 ----
		if(target.state != TrackerState::Tracking && target.state != TrackerState::TempLost)
		{
			LOG_DEBUG(kLogModule, "target state is not Tracking/TempLost");
			return solution;
		}

		if(!std::isfinite(t_now_s) || !std::isfinite(target.timestamp_s))
		{
			LOG_DEBUG(kLogModule, "non-finite timestamp");
			return solution;
		}

		if(t_now_s < target.timestamp_s)
		{
			LOG_DEBUG(kLogModule, "t_now_s < target.timestamp_s");
			return solution;
		}

		if(!finite_vehicle_state(target))
		{
			LOG_DEBUG(kLogModule, "non-finite vehicle state");
			return solution;
		}

		if(target.predicted_armors.empty() || !finite_hypotheses(target.predicted_armors))
		{
			LOG_DEBUG(kLogModule, "invalid predicted armors");
			return solution;
		}

		// ---- 状态作用域化（target_token 变化即同时清空 lock 与 predictive）----
		if(active_target_token_ != target.target_token)
		{
			lock_id_.reset();
			predictive_selected_armor_id_.reset();
			active_target_token_ = target.target_token;
		}

		// ---- transaction：捕获上一帧已提交 predictive 状态 ----
		const std::optional<int> previous_predictive_id = predictive_selected_armor_id_;

		if(debug != nullptr)
		{
			debug->previous_predictive_armor_id = previous_predictive_id;
		}

		// ---- 弹速处理（fallback / fail_safe）----
		const bool bullet_invalid =
		    !std::isfinite(bullet_speed_mps) || bullet_speed_mps < config_.min_valid_bullet_speed_mps;

		double bullet_speed = bullet_speed_mps;

		if(bullet_invalid)
		{
			if(config_.invalid_bullet_speed_policy == InvalidBulletSpeedPolicy::FailSafe)
			{
				solution.status = AimStatus::InvalidBulletSpeed;
				return solution;
			}

			bullet_speed = config_.fallback_bullet_speed_mps;
		}

		// ---- 延迟预测（signed yaw_rate 比较）----
		const double delay_time = (target.yaw_rate > config_.decision_speed_rad_s)
		                              ? config_.high_speed_delay_s
		                              : config_.low_speed_delay_s;

		const double t_muzzle = t_now_s + delay_time;

		// ---- 初始：预测到 t_muzzle，选板并求初始弹道 ----
		SolveResult current = solve_at_prediction_time(config_, lock_id_, previous_predictive_id,
		                                               target, bullet_speed, t_muzzle);

		if(!current.valid)
		{
			solution.status = current.status;
			return solution;
		}

		double prev_fly_time = current.fly_time;
		int refinement_iterations = 0;
		bool converged = false;

		// ---- 飞行时间迭代（最多 max_refinement_iterations 次）----
		for(int iter = 0; iter < config_.max_refinement_iterations; ++iter)
		{
			refinement_iterations = iter + 1;

			const double t_predict = t_muzzle + prev_fly_time;
			const SolveResult next = solve_at_prediction_time(
			    config_, lock_id_, previous_predictive_id, target, bullet_speed, t_predict);

			if(!next.valid)
			{
				solution.status = next.status;
				return solution;
			}

			current = next;

			if(std::abs(current.fly_time - prev_fly_time) < config_.flight_time_convergence_s)
			{
				converged = true;
				break;
			}

			prev_fly_time = current.fly_time;
		}

		// ---- 最终角度 ----
		solution.valid = true;
		solution.status = AimStatus::Success;
		solution.yaw_rad =
		    std::atan2(current.aim_point.y(), current.aim_point.x()) + config_.yaw_offset_rad;
		solution.pitch_rad = -(current.pitch + config_.pitch_offset_rad);
		solution.selected_armor_id = current.selected_armor_id;
		solution.fire_allowed = false; // 本阶段恒 false。

		// ---- transaction commit：仅成功时提交 predictive 状态 ----
		predictive_selected_armor_id_ = current.selected_armor_id;

		if(debug != nullptr)
		{
			debug->t_muzzle_s = t_muzzle;
			debug->target_prediction_time_s = current.target_prediction_time_s;
			debug->ballistic_arrival_time_s = t_muzzle + current.fly_time;
			debug->armor_selection_time_s = current.armor_selection_time_s;
			debug->switch_advance_s = current.switch_advance_s;
			debug->flight_time_s = current.fly_time;
			debug->aim_point_in_world = current.aim_point;
			debug->refinement_iterations = refinement_iterations;
			debug->ballistic_converged = converged;
			debug->pending_predictive_armor_id = current.selected_armor_id;
		}

		return solution;
	}

} // namespace app::auto_aim
