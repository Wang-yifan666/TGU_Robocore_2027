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

	} // namespace

	Aimer::Aimer(const AimerConfig& config): config_(config)
	{
		validate_aimer_config(config_);
	}

	void Aimer::reset()
	{
		lock_id_.reset();
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

		// ---- armor lock 作用域化（target_token 变化即清空）----
		if(active_target_token_ != target.target_token)
		{
			lock_id_.reset();
			active_target_token_ = target.target_token;
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
		const double dt_muzzle = t_muzzle - target.timestamp_s;
		const PredictedVehicle vehicle0 = predict_vehicle(target, dt_muzzle);

		if(!vehicle0.center.allFinite())
		{
			solution.status = AimStatus::PredictionFailed;
			return solution;
		}

		const std::vector<ArmorHypothesis> hypotheses0 = armor_hypotheses(vehicle0);
		const std::optional<ArmorHypothesis> aim0 =
		    select_armor(config_, lock_id_, vehicle0, hypotheses0, target.yaw_rate,
		                 target.has_armor_switch, target.name);

		if(!aim0)
		{
			solution.status = AimStatus::NoValidArmor;
			return solution;
		}

		const Eigen::Vector3d xyz0 = aim0->position_in_world;
		const double d0 = std::hypot(xyz0.x(), xyz0.y());

		tools::Trajectory trajectory0(bullet_speed, d0, xyz0.z());

		if(trajectory0.unsolvable)
		{
			solution.status = AimStatus::BallisticUnsolvable;
			return solution;
		}

		double prev_fly_time = trajectory0.fly_time;
		tools::Trajectory current_traj = trajectory0;
		Eigen::Vector3d aim_point = xyz0;
		std::optional<int> selected_armor_id = aim0->armor_id;

		int refinement_iterations = 0;
		bool converged = false;

		// ---- 飞行时间迭代（最多 max_refinement_iterations 次）----
		for(int iter = 0; iter < config_.max_refinement_iterations; ++iter)
		{
			refinement_iterations = iter + 1;

			const double dt_hit = (t_muzzle + prev_fly_time) - target.timestamp_s;
			const PredictedVehicle vehicle = predict_vehicle(target, dt_hit);

			if(!vehicle.center.allFinite())
			{
				solution.status = AimStatus::PredictionFailed;
				return solution;
			}

			const std::vector<ArmorHypothesis> hypotheses = armor_hypotheses(vehicle);
			const std::optional<ArmorHypothesis> aim =
			    select_armor(config_, lock_id_, vehicle, hypotheses, target.yaw_rate,
			                 target.has_armor_switch, target.name);

			if(!aim)
			{
				solution.status = AimStatus::NoValidArmor;
				return solution;
			}

			aim_point = aim->position_in_world;
			selected_armor_id = aim->armor_id;

			const double d = std::hypot(aim_point.x(), aim_point.y());
			current_traj = tools::Trajectory(bullet_speed, d, aim_point.z());

			if(current_traj.unsolvable)
			{
				solution.status = AimStatus::BallisticUnsolvable;
				return solution;
			}

			if(std::abs(current_traj.fly_time - prev_fly_time) < config_.flight_time_convergence_s)
			{
				converged = true;
				break;
			}

			prev_fly_time = current_traj.fly_time;
		}

		// ---- 最终角度 ----
		solution.valid = true;
		solution.status = AimStatus::Success;
		solution.yaw_rad = std::atan2(aim_point.y(), aim_point.x()) + config_.yaw_offset_rad;
		solution.pitch_rad = -(current_traj.pitch + config_.pitch_offset_rad);
		solution.selected_armor_id = selected_armor_id;
		solution.fire_allowed = false; // 本阶段恒 false。

		if(debug != nullptr)
		{
			debug->t_muzzle_s = t_muzzle;
			debug->t_hit_s = t_muzzle + current_traj.fly_time;
			debug->flight_time_s = current_traj.fly_time;
			debug->aim_point_in_world = aim_point;
			debug->refinement_iterations = refinement_iterations;
			debug->ballistic_converged = converged;
		}

		return solution;
	}

} // namespace app::auto_aim
