/**
 * @file tracker.cpp
 * @brief Tracker 生命周期状态机实现。
 */

#include "app/auto_aim/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace app::auto_aim
{

	namespace
	{
		constexpr double kSymmetryTolerance = 1e-9;

		bool finite_and_square_symmetric(const Eigen::MatrixXd& m, Eigen::Index dim)
		{
			if(m.rows() != dim || m.cols() != dim)
			{
				return false;
			}

			if(!m.allFinite())
			{
				return false;
			}

			const double scale = std::max(1.0, m.norm());
			return (m - m.transpose()).norm() <= kSymmetryTolerance * scale;
		}

		// 校验：正确 shape + finite + 近似对称 + positive-semidefinite（PSD）。
		// 只要求 PSD（λ_min >= -tolerance），不要求 positive definite：
		// singular covariance 是合法输入（例如测试用零协方差触发 correction 数值失败）。
		bool finite_square_symmetric_psd(const Eigen::MatrixXd& m, Eigen::Index dim)
		{
			if(!finite_and_square_symmetric(m, dim))
			{
				return false;
			}

			// SelfAdjointEigenSolver 只读取下三角；先对称化以消除近似对称带来的微小不对称。
			const Eigen::MatrixXd symmetric = 0.5 * (m + m.transpose());

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);

			if(solver.info() != Eigen::Success)
			{
				return false;
			}

			const Eigen::VectorXd& eigenvalues = solver.eigenvalues();

			// 数值 PSD：SelfAdjointEigenSolver 特征值误差上界约 O(eps * ||A||)，
			// 取 64 * eps * dim * max_abs_eigenvalue 作为 scale-aware tolerance，
			// 允许 roundoff 级别负特征值，同时拒绝真实负特征值。
			const double max_abs_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
			const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
			                         * static_cast<double>(dim) * max_abs_eigenvalue;

			return eigenvalues.minCoeff() >= -tolerance;
		}

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

	void validate_tracker_config(const TrackerConfig& config)
	{
		if(config.detecting_confirm_hits < 1)
		{
			throw std::invalid_argument("detecting_confirm_hits must be >= 1");
		}

		if(config.detecting_max_misses < 0)
		{
			throw std::invalid_argument("detecting_max_misses must be >= 0");
		}

		if(config.temp_lost_max_misses < 0)
		{
			throw std::invalid_argument("temp_lost_max_misses must be >= 0");
		}

		if(!std::isfinite(config.max_dt_s) || config.max_dt_s <= 0.0)
		{
			throw std::invalid_argument("max_dt_s must be finite and > 0");
		}

		const AssociationConfig& a = config.association;

		if(!std::isfinite(a.max_position_error_m) || a.max_position_error_m < 0.0)
		{
			throw std::invalid_argument("max_position_error_m must be finite and >= 0");
		}

		if(!std::isfinite(a.max_yaw_error_rad) || a.max_yaw_error_rad < 0.0)
		{
			throw std::invalid_argument("max_yaw_error_rad must be finite and >= 0");
		}

		if(!std::isfinite(a.position_score_scale_m) || a.position_score_scale_m <= 0.0)
		{
			throw std::invalid_argument("position_score_scale_m must be finite and > 0");
		}

		if(!std::isfinite(a.yaw_score_scale_rad) || a.yaw_score_scale_rad <= 0.0)
		{
			throw std::invalid_argument("yaw_score_scale_rad must be finite and > 0");
		}

		if(!finite_square_symmetric_psd(config.initial_covariance, kTargetStateDim))
		{
			throw std::invalid_argument(
			    "initial_covariance must be 11x11 finite symmetric PSD");
		}

		if(!finite_square_symmetric_psd(config.measurement_noise.base_covariance,
		                                kTargetMeasurementDim))
		{
			throw std::invalid_argument(
			    "measurement_noise.base_covariance must be 4x4 finite symmetric PSD");
		}

		if(!std::isfinite(config.measurement_noise.distance_angle_log_gain)
		   || config.measurement_noise.distance_angle_log_gain < 0.0)
		{
			throw std::invalid_argument("distance_angle_log_gain must be finite and >= 0");
		}

		if(!std::isfinite(config.measurement_noise.armor_yaw_distance_log_gain)
		   || config.measurement_noise.armor_yaw_distance_log_gain < 0.0)
		{
			throw std::invalid_argument("armor_yaw_distance_log_gain must be finite and >= 0");
		}

		const TargetModelConfig& p = config.process_noise;

		auto non_negative_finite = [](double v) {
			return std::isfinite(v) && v >= 0.0;
		};

		if(!non_negative_finite(p.translation_accel_variance)
		   || !non_negative_finite(p.yaw_accel_variance)
		   || !non_negative_finite(p.radius_random_walk_variance)
		   || !non_negative_finite(p.delta_radius_random_walk_variance)
		   || !non_negative_finite(p.delta_z_random_walk_variance))
		{
			throw std::invalid_argument(
			    "process noise variances must be finite and non-negative");
		}

		if(!std::isfinite(config.min_radius_m) || !std::isfinite(config.max_radius_m)
		   || config.min_radius_m <= 0.0 || config.max_radius_m < config.min_radius_m)
		{
			throw std::invalid_argument("radius bounds must satisfy 0 < min <= max");
		}

		const RadiusProfile& rp = config.radius_profile;

		auto positive_finite = [](double v) {
			return std::isfinite(v) && v > 0.0;
		};

		if(!positive_finite(rp.balance_2) || !positive_finite(rp.outpost_3)
		   || !positive_finite(rp.base_3) || !positive_finite(rp.default_4))
		{
			throw std::invalid_argument("radius profile values must be finite and > 0");
		}

		// 每个 RadiusProfile 值必须落在 [min_radius_m, max_radius_m] 之内。
		auto radius_in_range = [config](double v) {
			return v >= config.min_radius_m && v <= config.max_radius_m;
		};

		if(!radius_in_range(rp.balance_2) || !radius_in_range(rp.outpost_3)
		   || !radius_in_range(rp.base_3) || !radius_in_range(rp.default_4))
		{
			throw std::invalid_argument(
			    "radius profile values must lie within [min_radius_m, max_radius_m]");
		}
	}

	TrackerConfig make_default_tracker_config()
	{
		TrackerConfig c;

		c.detecting_confirm_hits = 3;
		c.detecting_max_misses = 3;
		c.temp_lost_max_misses = 10;
		c.max_dt_s = 0.5;

		c.association.max_position_error_m = 0.5;
		c.association.max_yaw_error_rad = 0.5;
		c.association.position_score_scale_m = 1.0;
		c.association.yaw_score_scale_rad = 1.0;

		c.initial_covariance = Eigen::MatrixXd::Identity(kTargetStateDim, kTargetStateDim);
		c.measurement_noise.base_covariance = Eigen::MatrixXd::Zero(
		    kTargetMeasurementDim, kTargetMeasurementDim);
		c.measurement_noise.base_covariance(0, 0) = 4e-3;  // bearing_yaw variance (rad^2)
		c.measurement_noise.base_covariance(1, 1) = 4e-3;  // pitch variance (rad^2)
		c.measurement_noise.base_covariance(2, 2) = 1.0;   // distance variance (m^2)
		c.measurement_noise.base_covariance(3, 3) = 9e-2;  // armor_yaw variance (rad^2)
		c.measurement_noise.distance_angle_log_gain = 1.0;
		c.measurement_noise.armor_yaw_distance_log_gain = 1.0 / 200.0;

		c.process_noise.translation_accel_variance = 1.0;
		c.process_noise.yaw_accel_variance = 1.0;
		c.process_noise.radius_random_walk_variance = 1.0;
		c.process_noise.delta_radius_random_walk_variance = 1.0;
		c.process_noise.delta_z_random_walk_variance = 1.0;

		c.min_radius_m = 0.05;
		c.max_radius_m = 0.5;

		c.radius_profile.balance_2 = 0.2;
		c.radius_profile.outpost_3 = 0.2765;
		c.radius_profile.base_3 = 0.3205;
		c.radius_profile.default_4 = 0.2;

		return c;
	}

	Tracker::Tracker(const TrackerConfig& config): config_(config)
	{
		validate_tracker_config(config);
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

	TrackResult Tracker::track(const std::vector<ArmorObservation>& observations,
	                           double timestamp_s)
	{
		if(!std::isfinite(timestamp_s))
		{
			throw std::invalid_argument("timestamp_s must be finite");
		}

		// ---- 尚无 target：初始化 ----
		if(state_ == TrackerState::Lost && !target_.has_value())
		{
			const std::optional<ArmorObservation> candidate = select_initial_observation(observations);

			if(!candidate)
			{
				return TrackResult{timestamp_s, TrackUpdateOutcome::NotTracked, std::nullopt};
			}

			const double radius = radius_for(*candidate);

			target_.emplace(*candidate, radius, config_.initial_covariance, config_.process_noise);
			hit_count_ = 1;
			miss_count_ = 0;

			// 初始化帧可能进入 Detecting（confirm_hits > 1）或 Tracking（confirm_hits == 1）。
			state_ = (hit_count_ >= config_.detecting_confirm_hits) ? TrackerState::Tracking
			                                                       : TrackerState::Detecting;

			last_timestamp_s_ = timestamp_s;
			has_timestamp_ = true;

			TrackedTarget snapshot = make_snapshot(state_, timestamp_s);
			return TrackResult{timestamp_s, TrackUpdateOutcome::Initialized, snapshot};
		}

		// ---- 已有 target ----
		const double dt = has_timestamp_ ? (timestamp_s - last_timestamp_s_) : 0.0;

		// dt < 0（timestamp regression）或 dt 过大：reset 后初始化当前帧。
		if(dt < 0.0 || dt > config_.max_dt_s)
		{
			reset();

			const std::optional<ArmorObservation> candidate =
			    select_initial_observation(observations);

			if(!candidate)
			{
				return TrackResult{timestamp_s, TrackUpdateOutcome::NotTracked, std::nullopt};
			}

			const double radius = radius_for(*candidate);

			target_.emplace(*candidate, radius, config_.initial_covariance, config_.process_noise);
			hit_count_ = 1;
			miss_count_ = 0;

			state_ = (hit_count_ >= config_.detecting_confirm_hits) ? TrackerState::Tracking
			                                                       : TrackerState::Detecting;

			last_timestamp_s_ = timestamp_s;
			has_timestamp_ = true;

			TrackedTarget snapshot = make_snapshot(state_, timestamp_s);
			return TrackResult{timestamp_s, TrackUpdateOutcome::Initialized, snapshot};
		}

		// 正常预测（dt == 0 允许，不除以 dt）。
		target_->predict(dt);
		last_timestamp_s_ = timestamp_s;

		// 记录 correction 前的先验预测中心（board-switch continuity 指标用）。
		// 注意：predict 之后即有效，即使本帧 NoAssociation 也会保留。
		const Eigen::VectorXd& predicted_state = target_->state();
		const Eigen::Vector3d prior_predicted_center(
		    predicted_state(kStateX), predicted_state(kStateY), predicted_state(kStateZ));

		// association + correction。
		const std::optional<AssociationResult> assoc =
		    associate(*target_, observations, config_.association);

		const bool associated = assoc.has_value();
		bool corrected = false;

		if(assoc)
		{
			const ArmorObservation& matched = observations[assoc->observation_index];
			corrected = target_->correct(matched, assoc->armor_id, config_.measurement_noise);
		}

		// 先结算本帧 outcome（与 target 生命周期分离；Lost 也保留真实结果）。
		TrackUpdateOutcome frame_outcome = TrackUpdateOutcome::NoAssociation;

		if(corrected)
		{
			frame_outcome = TrackUpdateOutcome::Corrected;
		}
		else if(associated)
		{
			frame_outcome = TrackUpdateOutcome::CorrectionFailed;
		}

		// geometry health：几何硬失败即 Lost（不属于 CorrectionFailed/NoAssociation 语义）。
		if(!geometry_healthy())
		{
			reset();
			return TrackResult{timestamp_s, TrackUpdateOutcome::NotTracked, std::nullopt};
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
				// miss（NoAssociation 或 CorrectionFailed）打断连续命中 streak：
				// hit_count_ 必须清零，重新积累 consecutive successful corrections。
				hit_count_ = 0;
				++miss_count_;

				if(miss_count_ > config_.detecting_max_misses)
				{
					reset();
					return TrackResult{timestamp_s, frame_outcome, std::nullopt};
				}
			}

			break;

		case TrackerState::Tracking:
			if(!corrected)
			{
				// temp_lost_max_misses == 0 时不输出 TempLost 帧，直接 Lost。
				if(config_.temp_lost_max_misses == 0)
				{
					reset();
					return TrackResult{timestamp_s, frame_outcome, std::nullopt};
				}

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
					return TrackResult{timestamp_s, frame_outcome, std::nullopt};
				}
			}

			break;

		case TrackerState::Lost:
		default:
			break;
		}

		TrackedTarget snapshot = make_snapshot(state_, timestamp_s);
		snapshot.prior_predicted_center = prior_predicted_center;

		if(corrected)
		{
			snapshot.has_measurement = true;
			snapshot.matched_observation_index = assoc->observation_index;
			snapshot.matched_armor_id = assoc->armor_id;
			snapshot.innovation = target_->last_innovation();
			snapshot.nis = target_->last_nis();
		}

		return TrackResult{timestamp_s, frame_outcome, snapshot};
	}

} // namespace app::auto_aim