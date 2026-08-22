/**
 * @file association.cpp
 * @brief Target 与 ArmorObservation 的确定性关联实现。
 */

#include "app/auto_aim/association.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace app::auto_aim
{

	namespace
	{

		bool unknown_compatible(ArmorColor lhs, ArmorColor rhs)
		{
			return lhs == ArmorColor::Unknown || rhs == ArmorColor::Unknown || lhs == rhs;
		}

		// identity 兼容：name/type 必须一致；color 两者非 Unknown 时必须一致。
		bool identity_compatible(const Target& target, const ArmorObservation& observation)
		{
			if(target.name() != observation.name)
			{
				return false;
			}

			if(target.type() != observation.type)
			{
				return false;
			}

			return unknown_compatible(target.color(), observation.color);
		}

	} // namespace

	std::optional<AssociationResult> associate(const Target& target,
	                                           const std::vector<ArmorObservation>& observations,
	                                           const AssociationConfig& config)
	{
		if(config.max_position_error_m < 0.0 || !std::isfinite(config.max_position_error_m))
		{
			throw std::invalid_argument("max_position_error_m must be finite and >= 0");
		}

		if(config.max_yaw_error_rad < 0.0 || !std::isfinite(config.max_yaw_error_rad))
		{
			throw std::invalid_argument("max_yaw_error_rad must be finite and >= 0");
		}

		if(!std::isfinite(config.position_score_scale_m) || config.position_score_scale_m <= 0.0)
		{
			throw std::invalid_argument("position_score_scale_m must be finite and > 0");
		}

		if(!std::isfinite(config.yaw_score_scale_rad) || config.yaw_score_scale_rad <= 0.0)
		{
			throw std::invalid_argument("yaw_score_scale_rad must be finite and > 0");
		}

		const std::vector<ArmorHypothesis> hypotheses = target.armor_hypotheses();

		std::optional<AssociationResult> best;

		for(std::size_t j = 0; j < observations.size(); ++j)
		{
			const ArmorObservation& observation = observations[j];

			if(!observation.position_in_world.allFinite() || !std::isfinite(observation.armor_yaw_in_world))
			{
				continue;
			}

			if(!identity_compatible(target, observation))
			{
				continue;
			}

			for(const ArmorHypothesis& hypothesis: hypotheses)
			{
				const Eigen::Vector3d position_error =
				    observation.position_in_world - hypothesis.position_in_world;

				const double position_norm = position_error.norm();

				if(!std::isfinite(position_norm))
				{
					continue;
				}

				if(position_norm > config.max_position_error_m)
				{
					continue;
				}

				const double yaw_error =
				    wrap_angle(observation.armor_yaw_in_world - hypothesis.yaw_in_world);

				if(std::abs(yaw_error) > config.max_yaw_error_rad)
				{
					continue;
				}

				const double score =
				    position_error.squaredNorm() / (config.position_score_scale_m
				                                    * config.position_score_scale_m)
				    + (yaw_error * yaw_error) / (config.yaw_score_scale_rad
				                                 * config.yaw_score_scale_rad);

				AssociationResult candidate;
				candidate.observation_index = j;
				candidate.armor_id = hypothesis.armor_id;
				candidate.score = score;
				candidate.position_residual = position_error;
				candidate.yaw_residual = yaw_error;

				if(!best)
				{
					best = candidate;
					continue;
				}

				// 确定性 tie-break：score 小者优先；observation index 小者；armor_id 小者。
				bool replace = false;
				if(candidate.score < best->score)
				{
					replace = true;
				}
				else if(candidate.score == best->score
				        && candidate.observation_index < best->observation_index)
				{
					replace = true;
				}
				else if(candidate.score == best->score
				        && candidate.observation_index == best->observation_index
				        && candidate.armor_id < best->armor_id)
				{
					replace = true;
				}

				if(replace)
				{
					best = candidate;
				}
			}
		}

		return best;
	}

} // namespace app::auto_aim