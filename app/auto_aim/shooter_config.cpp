/**
 * @file shooter_config.cpp
 * @brief Shooter 配置加载实现。
 */

#include "app/auto_aim/shooter_config.hpp"

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
		constexpr std::string_view kLogModule = "SHOOTER_CONFIG";

		std::optional<double> read_double(const toml::table& table, std::string_view key)
		{
			return table[key].value<double>();
		}

		std::optional<bool> read_bool(const toml::table& table, std::string_view key)
		{
			return table[key].value<bool>();
		}
	} // namespace

	bool load_shooter_config_from_table(const toml::table& root, ShooterConfig& config)
	{
		const auto* shooter_table = root["shooter"].as_table();

		if(shooter_table == nullptr)
		{
			LOG_ERROR(kLogModule, "missing required table: shooter");
			return false;
		}

		ShooterConfig loaded;

		// auto_fire（required bool）。
		{
			const auto v = read_bool(*shooter_table, "auto_fire");

			if(!v)
			{
				LOG_ERROR(kLogModule, "auto_fire must be a boolean");
				return false;
			}

			loaded.auto_fire = *v;
		}

		// near_tolerance_rad。
		{
			const auto v = read_double(*shooter_table, "near_tolerance_rad");

			if(!v || !std::isfinite(*v) || *v < 0.0)
			{
				LOG_ERROR(kLogModule, "near_tolerance_rad must be finite and >= 0");
				return false;
			}

			loaded.near_tolerance_rad = *v;
		}

		// far_tolerance_rad。
		{
			const auto v = read_double(*shooter_table, "far_tolerance_rad");

			if(!v || !std::isfinite(*v) || *v < 0.0)
			{
				LOG_ERROR(kLogModule, "far_tolerance_rad must be finite and >= 0");
				return false;
			}

			loaded.far_tolerance_rad = *v;
		}

		// distance_threshold_m。
		{
			const auto v = read_double(*shooter_table, "distance_threshold_m");

			if(!v || !std::isfinite(*v) || *v < 0.0)
			{
				LOG_ERROR(kLogModule, "distance_threshold_m must be finite and >= 0");
				return false;
			}

			loaded.distance_threshold_m = *v;
		}

		try
		{
			validate_shooter_config(loaded);
		}
		catch(const std::exception& exception)
		{
			LOG_ERROR(kLogModule, "shooter configuration validation failed: {}", exception.what());
			return false;
		}

		config = std::move(loaded);
		return true;
	}

	bool load_shooter_config(const std::string& config_path, ShooterConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			if(!load_shooter_config_from_table(root, config))
			{
				LOG_ERROR(kLogModule, "shooter configuration validation failed: {}", config_path);
				return false;
			}

			LOG_INFO(kLogModule, "loaded shooter config: {}", config_path);
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
