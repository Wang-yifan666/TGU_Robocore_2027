/**
 * @file shooter.cpp
 * @brief Shooter 实现。
 */

#include "app/auto_aim/shooter.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tools/logger.hpp"
#include "tools/maths_tools.hpp"

namespace app::auto_aim
{

	namespace
	{
		constexpr std::string_view kLogModule = "SHOOTER";
	} // namespace

	void validate_shooter_config(const ShooterConfig& config)
	{
		const auto require_non_negative = [](double value, const char* name) {
			if(!std::isfinite(value) || value < 0.0)
			{
				throw std::invalid_argument(std::string(name) + " must be finite and >= 0");
			}
		};

		require_non_negative(config.near_tolerance_rad, "near_tolerance_rad");
		require_non_negative(config.far_tolerance_rad, "far_tolerance_rad");
		require_non_negative(config.distance_threshold_m, "distance_threshold_m");
	}

	ShooterConfig make_default_shooter_config()
	{
		ShooterConfig config;

		// fail-safe 默认关闭；生产 TOML 显式开启。
		config.auto_fire = false;

		// 容差保留 SP25 的 /57.3 近似，与 aimer.toml 的 coming_angle 等字段数值口径一致。
		config.near_tolerance_rad = 5.0 / 57.3; // SP25 first_tolerance（5°）
		config.far_tolerance_rad = 2.0 / 57.3;  // SP25 second_tolerance（2°）
		config.distance_threshold_m = 3.0;      // SP25 judge_distance

		return config;
	}

	Shooter::Shooter(const ShooterConfig& config): config_(config)
	{
		validate_shooter_config(config);

		if(!config_.auto_fire)
		{
			LOG_INFO(kLogModule, "auto fire disabled");
		}
	}

	void Shooter::reset()
	{
		previous_aiming_yaw_rad_.reset();
	}

	bool Shooter::shoot(const AimingSolution& aiming, double target_distance_m,
	                    double gimbal_yaw_rad)
	{
		// 1) auto_fire 关闭（正常配置状态，非异常）。
		if(!config_.auto_fire)
		{
			LOG_DEBUG(kLogModule, "auto fire disabled");
			return false;
		}

		// 2) aiming invalid（正常上游失败，防御分支）。
		if(!aiming.valid)
		{
			LOG_DEBUG(kLogModule, "invalid aiming solution");
			return false;
		}

		// 3) gimbal NaN 是 missing-feedback sentinel（未接云台反馈），使用 DEBUG。
		// 丢失时清空历史，使反馈恢复后的第一帧重新建立历史并禁止开火。
		if(!std::isfinite(gimbal_yaw_rad))
		{
			previous_aiming_yaw_rad_.reset();
			LOG_DEBUG(kLogModule, "gimbal feedback missing (NaN), reset history, no fire");
			return false;
		}

		// 4) 其余非有限输入（yaw/pitch/distance）：异常，fail-safe。
		if(!std::isfinite(aiming.yaw_rad) || !std::isfinite(aiming.pitch_rad)
		   || !std::isfinite(target_distance_m))
		{
			LOG_WARN(kLogModule, "non-finite input: yaw={} pitch={} dist={}", aiming.yaw_rad,
			         aiming.pitch_rad, target_distance_m);
			return false;
		}

		// 5) 负距离：上游语义错误，fail-safe。
		if(target_distance_m < 0.0)
		{
			LOG_WARN(kLogModule, "negative target distance: {}", target_distance_m);
			return false;
		}

		// 6) 距离选档：> 阈值用远距更严 tolerance。
		const double tolerance = (target_distance_m > config_.distance_threshold_m)
		    ? config_.far_tolerance_rad
		    : config_.near_tolerance_rad;

		// 7) 首帧：建立历史，禁止开火。
		if(!previous_aiming_yaw_rad_.has_value())
		{
			previous_aiming_yaw_rad_ = aiming.yaw_rad;
			LOG_DEBUG(kLogModule, "first valid aim, establish history, no fire");
			return false;
		}

		const double prev_yaw = *previous_aiming_yaw_rad_;

		// 8) 周期角 wrap 后比较（复用 tools::maths_tools::limit_rad）。
		const double aim_jump = std::abs(tools::maths_tools::limit_rad(aiming.yaw_rad - prev_yaw));
		const double gimbal_lag = std::abs(tools::maths_tools::limit_rad(gimbal_yaw_rad - prev_yaw));

		const bool stable = aim_jump < 2.0 * tolerance;
		const bool settled = gimbal_lag < tolerance;

		// 9) 有效帧无论是否开火都更新历史（SP25 语义）。
		previous_aiming_yaw_rad_ = aiming.yaw_rad;

		if(stable && settled)
		{
			LOG_DEBUG(kLogModule, "fire allowed");
			return true;
		}

		LOG_DEBUG(kLogModule, "fire denied: stable={} settled={} jump={:.4f} lag={:.4f} tol={:.4f}",
		          stable, settled, aim_jump, gimbal_lag, tolerance);
		return false;
	}

} // namespace app::auto_aim
