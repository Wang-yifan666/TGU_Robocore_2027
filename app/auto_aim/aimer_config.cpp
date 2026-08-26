/**
 * @file aimer_config.cpp
 * @brief Aimer 配置加载实现。
 */

#include "app/auto_aim/aimer_config.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "tools/logger.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "AIMER_CONFIG";

		std::optional<double> read_double(const toml::table& table, std::string_view key)
		{
			return table[key].value<double>();
		}

		std::optional<bool> read_bool(const toml::table& table, std::string_view key)
		{
			return table[key].value<bool>();
		}

		std::optional<std::int64_t> read_int(const toml::table& table, std::string_view key)
		{
			return table[key].value<std::int64_t>();
		}

		std::optional<std::string> read_string(const toml::table& table, std::string_view key)
		{
			return table[key].value<std::string>();
		}

	} // namespace

	AimerConfig make_default_aimer_config()
	{
		AimerConfig c;

		c.yaw_offset_rad = 0.0;
		c.pitch_offset_rad = 0.0;

		// SP25 使用 /57.3 做 degree -> rad，默认值保留该因子以保持数值一致。
		c.coming_angle_rad = 60.0 / 57.3;
		c.leaving_angle_rad = 20.0 / 57.3;

		c.outpost_coming_angle_rad = 70.0 / 57.3;
		c.outpost_leaving_angle_rad = 30.0 / 57.3;

		c.shootable_angle_threshold_rad = 60.0 / 57.3;

		c.high_speed_delay_s = 0.0;
		c.low_speed_delay_s = 0.0;
		c.decision_speed_rad_s = 10.0;

		c.use_radius_for_gyro_detection = true;
		c.non_gyro_radius_threshold_m = 2.0;
		c.non_gyro_yaw_rate_threshold_rad_s = 2.0;

		c.invalid_bullet_speed_policy = InvalidBulletSpeedPolicy::Fallback;
		c.min_valid_bullet_speed_mps = 14.0;
		c.fallback_bullet_speed_mps = 23.0;

		c.max_refinement_iterations = 10;
		c.flight_time_convergence_s = 0.001;

		c.armor_switch_strategy = ArmorSwitchStrategy::SpCompat;

		return c;
	}

	void validate_aimer_config(const AimerConfig& config)
	{
		const auto require_finite = [](double value, const char* name) {
			if(!std::isfinite(value))
			{
				throw std::invalid_argument(std::string(name) + " must be finite");
			}
		};

		const auto require_non_negative = [&](double value, const char* name) {
			require_finite(value, name);

			if(value < 0.0)
			{
				throw std::invalid_argument(std::string(name) + " must be >= 0");
			}
		};

		require_finite(config.yaw_offset_rad, "yaw_offset_rad");
		require_finite(config.pitch_offset_rad, "pitch_offset_rad");
		require_non_negative(config.coming_angle_rad, "coming_angle_rad");
		require_non_negative(config.leaving_angle_rad, "leaving_angle_rad");
		require_non_negative(config.outpost_coming_angle_rad, "outpost_coming_angle_rad");
		require_non_negative(config.outpost_leaving_angle_rad, "outpost_leaving_angle_rad");
		require_non_negative(config.shootable_angle_threshold_rad, "shootable_angle_threshold_rad");

		require_non_negative(config.high_speed_delay_s, "high_speed_delay_s");
		require_non_negative(config.low_speed_delay_s, "low_speed_delay_s");
		require_non_negative(config.decision_speed_rad_s, "decision_speed_rad_s");

		require_non_negative(config.non_gyro_radius_threshold_m, "non_gyro_radius_threshold_m");
		require_non_negative(config.non_gyro_yaw_rate_threshold_rad_s,
		                     "non_gyro_yaw_rate_threshold_rad_s");

		if(!std::isfinite(config.min_valid_bullet_speed_mps) || config.min_valid_bullet_speed_mps <= 0.0)
		{
			throw std::invalid_argument("min_valid_bullet_speed_mps must be finite and > 0");
		}

		if(!std::isfinite(config.fallback_bullet_speed_mps) || config.fallback_bullet_speed_mps <= 0.0)
		{
			throw std::invalid_argument("fallback_bullet_speed_mps must be finite and > 0");
		}

		if(config.max_refinement_iterations < 1)
		{
			throw std::invalid_argument("max_refinement_iterations must be >= 1");
		}

		if(!std::isfinite(config.flight_time_convergence_s) || config.flight_time_convergence_s <= 0.0)
		{
			throw std::invalid_argument("flight_time_convergence_s must be finite and > 0");
		}
	}

	bool load_aimer_config_from_table(const toml::table& root, AimerConfig& config)
	{
		const auto* aimer_table = root["aimer"].as_table();

		if(aimer_table == nullptr)
		{
			LOG_ERROR(kLogModule, "missing required [aimer] table");
			return false;
		}

		AimerConfig loaded;

		// 角度（可为负）。
		const auto read_angle = [&](std::string_view key, double& out) {
			auto v = read_double(*aimer_table, key);

			if(!v || !std::isfinite(*v))
			{
				LOG_ERROR(kLogModule, "{} must be finite", key);
				return false;
			}

			out = *v;
			return true;
		};

		// 非负标量。
		const auto read_non_negative = [&](std::string_view key, double& out) {
			auto v = read_double(*aimer_table, key);

			if(!v || !std::isfinite(*v) || *v < 0.0)
			{
				LOG_ERROR(kLogModule, "{} must be finite and >= 0", key);
				return false;
			}

			out = *v;
			return true;
		};

		if(!read_angle("yaw_offset_rad", loaded.yaw_offset_rad)
		   || !read_angle("pitch_offset_rad", loaded.pitch_offset_rad)
		   || !read_non_negative("coming_angle_rad", loaded.coming_angle_rad)
		   || !read_non_negative("leaving_angle_rad", loaded.leaving_angle_rad)
		   || !read_non_negative("outpost_coming_angle_rad", loaded.outpost_coming_angle_rad)
		   || !read_non_negative("outpost_leaving_angle_rad", loaded.outpost_leaving_angle_rad)
		   || !read_non_negative("shootable_angle_threshold_rad",
		                         loaded.shootable_angle_threshold_rad)
		   || !read_non_negative("high_speed_delay_s", loaded.high_speed_delay_s)
		   || !read_non_negative("low_speed_delay_s", loaded.low_speed_delay_s)
		   || !read_non_negative("decision_speed_rad_s", loaded.decision_speed_rad_s)
		   || !read_non_negative("non_gyro_radius_threshold_m",
		                         loaded.non_gyro_radius_threshold_m)
		   || !read_non_negative("non_gyro_yaw_rate_threshold_rad_s",
		                         loaded.non_gyro_yaw_rate_threshold_rad_s))
		{
			return false;
		}

		// use_radius_for_gyro_detection（bool）。
		{
			auto v = read_bool(*aimer_table, "use_radius_for_gyro_detection");

			if(!v)
			{
				LOG_ERROR(kLogModule, "use_radius_for_gyro_detection must be a bool");
				return false;
			}

			loaded.use_radius_for_gyro_detection = *v;
		}

		// 弹速（> 0）。
		{
			auto min_v = read_double(*aimer_table, "min_valid_bullet_speed_mps");
			auto fallback_v = read_double(*aimer_table, "fallback_bullet_speed_mps");

			if(!min_v || !std::isfinite(*min_v) || *min_v <= 0.0)
			{
				LOG_ERROR(kLogModule, "min_valid_bullet_speed_mps must be finite and > 0");
				return false;
			}

			if(!fallback_v || !std::isfinite(*fallback_v) || *fallback_v <= 0.0)
			{
				LOG_ERROR(kLogModule, "fallback_bullet_speed_mps must be finite and > 0");
				return false;
			}

			loaded.min_valid_bullet_speed_mps = *min_v;
			loaded.fallback_bullet_speed_mps = *fallback_v;
		}

		// invalid_bullet_speed_policy（enum string）。
		{
			auto v = read_string(*aimer_table, "invalid_bullet_speed_policy");

			if(!v)
			{
				LOG_ERROR(kLogModule, "invalid_bullet_speed_policy must be a string");
				return false;
			}

			if(*v == "fallback")
			{
				loaded.invalid_bullet_speed_policy = InvalidBulletSpeedPolicy::Fallback;
			}
			else if(*v == "fail_safe")
			{
				loaded.invalid_bullet_speed_policy = InvalidBulletSpeedPolicy::FailSafe;
			}
			else
			{
				LOG_ERROR(kLogModule,
				          "invalid_bullet_speed_policy must be 'fallback' or 'fail_safe'");
				return false;
			}
		}

		// max_refinement_iterations。
		{
			auto iters = read_int(*aimer_table, "max_refinement_iterations");

			if(!iters || *iters < 1)
			{
				LOG_ERROR(kLogModule, "max_refinement_iterations must be an integer >= 1");
				return false;
			}

			loaded.max_refinement_iterations = static_cast<int>(*iters);
		}

		// flight_time_convergence_s。
		{
			auto conv = read_double(*aimer_table, "flight_time_convergence_s");

			if(!conv || !std::isfinite(*conv) || *conv <= 0.0)
			{
				LOG_ERROR(kLogModule, "flight_time_convergence_s must be finite and > 0");
				return false;
			}

			loaded.flight_time_convergence_s = *conv;
		}

		// armor_switch_strategy（enum string）。
		{
			auto v = read_string(*aimer_table, "armor_switch_strategy");

			if(!v)
			{
				LOG_ERROR(kLogModule, "armor_switch_strategy must be a string");
				return false;
			}

			if(*v == "sp_compat")
			{
				loaded.armor_switch_strategy = ArmorSwitchStrategy::SpCompat;
			}
			else
			{
				LOG_ERROR(kLogModule, "armor_switch_strategy must be 'sp_compat'");
				return false;
			}
		}

		try
		{
			validate_aimer_config(loaded);
		}
		catch(const std::exception& exception)
		{
			LOG_ERROR(kLogModule, "aimer configuration validation failed: {}", exception.what());
			return false;
		}

		config = std::move(loaded);
		return true;
	}

	bool load_aimer_config(const std::string& config_path, AimerConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			if(!load_aimer_config_from_table(root, config))
			{
				LOG_ERROR(kLogModule, "aimer configuration validation failed: {}", config_path);
				return false;
			}

			LOG_INFO(kLogModule, "loaded aimer config: {}", config_path);
			return true;
		}
		catch(const toml::parse_error& exception)
		{
			LOG_ERROR(kLogModule, "failed to parse {}: {}", config_path, exception.what());
			return false;
		}
		catch(const std::exception& exception)
		{
			LOG_ERROR(kLogModule, "failed to load {}: {}", config_path, exception.what());
			return false;
		}
		catch(...)
		{
			LOG_ERROR(kLogModule, "failed to load {}: unknown exception", config_path);
			return false;
		}
	}

} // namespace app::auto_aim
