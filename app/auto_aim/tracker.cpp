/**
 * @file tracker.cpp
 * @brief Tracker 生命周期状态机实现。
 */

#include "app/auto_aim/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace app::auto_aim
{

	namespace
	{
		int priority_rank(ArmorPriority priority)
		{
			switch(priority)
			{
			case ArmorPriority::First:
				return 0;
			case ArmorPriority::Second:
				return 1;
			case ArmorPriority::Third:
				return 2;
			case ArmorPriority::Fourth:
				return 3;
			case ArmorPriority::Fifth:
				return 4;
			case ArmorPriority::Unknown:
			default:
				return 5;
			}
		}
	} // namespace

	Tracker::Tracker(const TrackerConfig& config): config_(config)
	{
		if(config.detecting_confirm_hits < 0 || config.detecting_max_misses < 0
		   || config.temp_lost_max_misses < 0)
		{
			throw std::invalid_argument("hit/miss thresholds must be non-negative");
		}

		if(!std::isfinite(config.max_dt_s) || config.max_dt_s < 0.0)
		{
			throw std::invalid_argument("max_dt_s must be finite and >= 0");
		}

		if(!std::isfinite(config.min_radius_m) || !std::isfinite(config.max_radius_m)
		   || config.min_radius_m <= 0.0 || config.max_radius_m < config.min_radius_m)
		{
			throw std::invalid_argument("radius bounds must be 0 < min <= max");
		}
	}

	double Tracker::radius_for(const ArmorObservation& observation) const
	{
		const int count = armor_count_for(observation.type, observation.name);

		if(count == 2)
		{
			return config_.radius_profile.balance_2;
		}

		if(observation.name == ArmorName::Outpost)
		{
			return config_.radius_profile.outpost_3;
		}

		if(observation.name == ArmorName::Base)
		{
			return config_.radius_profile.base_3;
		}

		return config_.radius_profile.default_4;
	}

	std::optional<ArmorObservation> Tracker::select_initial_observation(
	    const std::vector<ArmorObservation>& observations) const
	{
		std::vector<std::size_t> indices(observations.size());
		std::iota(indices.begin(), indices.end(), std::size_t{0});

		// 过滤 invalid observation。
		indices.erase(
		    std::remove_if(indices.begin(), indices.end(),
		                   [&](std::size_t i) {
			                   return !observations[i].position_in_world.allFinite()
			                       || !std::isfinite(observations[i].armor_yaw_in_world);
		                   }),
		    indices.end());

		if(indices.empty())
		{
			return std::nullopt;
		}

		std::sort(indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
			const ArmorObservation& a = observations[lhs];
			const ArmorObservation& b = observations[rhs];

			const int pa = priority_rank(a.priority);
			const int pb = priority_rank(b.priority);

			if(pa != pb)
			{
				return pa < pb;
			}

			const double da = a.position_in_world.norm();
			const double db = b.position_in_world.norm();

			if(da != db)
			{
				return da < db;
			}

			if(a.source_detection_index != b.source_detection_index)
			{
				return a.source_detection_index < b.source_detection_index;
			}

			return lhs < rhs;
		});

		return observations[indices.front()];
	}

	bool Tracker::geometry_healthy() const
	{
		const Eigen::VectorXd& x = target_->state();

		const double radius = x(kStateRadius);
		const double alternate = x(kStateRadius) + x(kStateDeltaRadius);

		if(!std::isfinite(radius) || !std::isfinite(alternate))
		{
			return false;
		}

		return radius >= config_.min_radius_m && radius <= config_.max_radius_m
		    && alternate >= config_.min_radius_m && alternate <= config_.max_radius_m;
	}

	TrackedTarget Tracker::make_snapshot(TrackerState state, double timestamp_s) const
	{
		const Eigen::VectorXd& x = target_->state();

		TrackedTarget snapshot;
		snapshot.state = state;
		snapshot.has_measurement = false;
		snapshot.timestamp_s = timestamp_s;

		snapshot.color = target_->color();
		snapshot.name = target_->name();
		snapshot.type = target_->type();
		snapshot.priority = target_->priority();

		snapshot.center_in_world =
		    Eigen::Vector3d(x(kStateX), x(kStateY), x(kStateZ));
		snapshot.velocity_in_world =
		    Eigen::Vector3d(x(kStateVx), x(kStateVy), x(kStateVz));

		snapshot.yaw = x(kStateYaw);
		snapshot.yaw_rate = x(kStateYawRate);
		snapshot.radius = x(kStateRadius);
		snapshot.delta_radius = x(kStateDeltaRadius);
		snapshot.delta_z = x(kStateDeltaZ);

		snapshot.predicted_armors = target_->armor_hypotheses();

		return snapshot;
	}

	void Tracker::reset()
	{
		state_ = TrackerState::Lost;
		target_.reset();
		hit_count_ = 0;
		miss_count_ = 0;
		has_timestamp_ = false;
		last_timestamp_s_ = 0.0;
	}

	std::optional<TrackedTarget> Tracker::track(const std::vector<ArmorObservation>& observations,
	                                            double timestamp_s)
	{
		if(!std::isfinite(timestamp_s))
		{
			throw std::invalid_argument("timestamp_s must be finite");
		}

		// ---- 尚未有 target（Lost）----
		if(state_ == TrackerState::Lost && !target_.has_value())
		{
			const std::optional<ArmorObservation> candidate = select_initial_observation(observations);

			if(!candidate)
			{
				return std::nullopt;
			}

			const double radius = radius_for(*candidate);

			target_.emplace(*candidate, radius, config_.initial_covariance, config_.process_noise);
			state_ = TrackerState::Detecting;
			hit_count_ = 1;
			miss_count_ = 0;

			last_timestamp_s_ = timestamp_s;
			has_timestamp_ = true;

			return make_snapshot(TrackerState::Detecting, timestamp_s);
		}

		// ---- 已有 target ----
		const double dt = has_timestamp_ ? (timestamp_s - last_timestamp_s_) : 0.0;

		// dt < 0（timestamp regression）或 dt 过大：reset 后重新初始化当前帧。
		if(dt < 0.0 || dt > config_.max_dt_s)
		{
			reset();

			const std::optional<ArmorObservation> candidate =
			    select_initial_observation(observations);

			if(!candidate)
			{
				return std::nullopt;
			}

			const double radius = radius_for(*candidate);

			target_.emplace(*candidate, radius, config_.initial_covariance, config_.process_noise);
			state_ = TrackerState::Detecting;
			hit_count_ = 1;
			miss_count_ = 0;

			last_timestamp_s_ = timestamp_s;
			has_timestamp_ = true;

			return make_snapshot(TrackerState::Detecting, timestamp_s);
		}

		// 正常预测（dt == 0 允许，不除以 dt）。
		target_->predict(dt);
		last_timestamp_s_ = timestamp_s;

		// association + correction。
		const std::optional<AssociationResult> assoc =
		    associate(*target_, observations, config_.association);

		bool corrected = false;

		if(assoc)
		{
			const ArmorObservation& matched = observations[assoc->observation_index];
			corrected = target_->correct(matched, assoc->armor_id, config_.measurement_covariance);
		}

		// geometry health：失败即 Lost。
		if(!geometry_healthy())
		{
			reset();
			return std::nullopt;
		}

		// 状态机推进。
		switch(state_)
		{
		case TrackerState::Detecting:
			if(corrected)
			{
				++hit_count_;
				miss_count_ = 0;

				if(hit_count_ >= config_.detecting_confirm_hits)
				{
					state_ = TrackerState::Tracking;
				}
			}
			else
			{
				++miss_count_;

				if(miss_count_ > config_.detecting_max_misses)
				{
					reset();
					return std::nullopt;
				}
			}

			break;

		case TrackerState::Tracking:
			if(!corrected)
			{
				state_ = TrackerState::TempLost;
				miss_count_ = 1;
			}

			break;

		case TrackerState::TempLost:
			if(corrected)
			{
				state_ = TrackerState::Tracking;
				miss_count_ = 0;
			}
			else
			{
				++miss_count_;

				if(miss_count_ > config_.temp_lost_max_misses)
				{
					reset();
					return std::nullopt;
				}
			}

			break;

		case TrackerState::Lost:
		default:
			break;
		}

		TrackedTarget snapshot = make_snapshot(state_, timestamp_s);

		if(corrected && assoc)
		{
			snapshot.has_measurement = true;
			snapshot.matched_observation_index = assoc->observation_index;
			snapshot.matched_armor_id = assoc->armor_id;
			snapshot.innovation = target_->last_innovation();
			snapshot.nis = target_->last_nis();
		}

		return snapshot;
	}

} // namespace app::auto_aim