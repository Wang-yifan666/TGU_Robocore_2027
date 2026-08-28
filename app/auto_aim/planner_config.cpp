/**
 * @file planner_config.cpp
 * @brief Planner 配置加载实现。
 */

#include "app/auto_aim/planner_config.hpp"

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

		constexpr std::string_view kLogModule = "PLANNER_CONFIG";

		std::optional<double> read_double(const toml::table& table, std::string_view key)
		{
			return table[key].value<double>();
		}

		std::optional<std::int64_t> read_int(const toml::table& table, std::string_view key)
		{
			return table[key].value<std::int64_t>();
		}

		bool read_vector2(const toml::table& table, std::string_view key, Eigen::Vector2d& out)
		{
			const auto view = table[key];
			const toml::array* arr = view.as_array();

			if(arr == nullptr || arr->size() != 2)
			{
				return false;
			}

			const auto e0 = (*arr)[0].value<double>();
			const auto e1 = (*arr)[1].value<double>();

			if(!e0 || !e1)
			{
				return false;
			}

			out = Eigen::Vector2d(*e0, *e1);
			return true;
		}

	} // namespace

	PlannerConfig make_default_planner_config()
	{
		PlannerConfig config;

		// legacy standard4.yaml：max_yaw_acc=50, max_pitch_acc=100, Q=[9e6,0], R=1, rho=1, max_iter=10。
		config.max_yaw_acceleration_rad_s2 = 50.0;
		config.max_pitch_acceleration_rad_s2 = 100.0;
		config.q_yaw = Eigen::Vector2d(9e6, 0.0);
		config.r_yaw = 1.0;
		config.q_pitch = Eigen::Vector2d(9e6, 0.0);
		config.r_pitch = 1.0;
		config.rho = 1.0;
		config.max_iter = 10;

		return config;
	}

	void validate_planner_config(const PlannerConfig& config)
	{
		const auto require_positive = [](double value, const char* name) {
			if(!std::isfinite(value) || value <= 0.0)
			{
				throw std::invalid_argument(std::string(name) + " must be finite and > 0");
			}
		};
		const auto require_semidefinite = [](const Eigen::Vector2d& q, const char* name) {
			if(!q.allFinite() || q.minCoeff() < 0.0)
			{
				throw std::invalid_argument(std::string(name) + " must be finite and >= 0");
			}
		};

		require_positive(config.max_yaw_acceleration_rad_s2, "max_yaw_acceleration_rad_s2");
		require_positive(config.max_pitch_acceleration_rad_s2, "max_pitch_acceleration_rad_s2");
		require_semidefinite(config.q_yaw, "q_yaw");
		require_positive(config.r_yaw, "r_yaw");
		require_semidefinite(config.q_pitch, "q_pitch");
		require_positive(config.r_pitch, "r_pitch");
		require_positive(config.rho, "rho");

		if(config.max_iter < 1)
		{
			throw std::invalid_argument("max_iter must be >= 1");
		}
	}

	bool load_planner_config_from_table(const toml::table& root, PlannerConfig& config)
	{
		const auto* planner_table = root["planner"].as_table();

		if(planner_table == nullptr)
		{
			LOG_ERROR(kLogModule, "missing required table: planner");
			return false;
		}

		PlannerConfig loaded;

		if(const auto v = read_double(*planner_table, "max_yaw_acceleration_rad_s2"))
		{
			loaded.max_yaw_acceleration_rad_s2 = *v;
		}
		else
		{
			LOG_ERROR(kLogModule, "max_yaw_acceleration_rad_s2 must be a finite double");
			return false;
		}

		if(const auto v = read_double(*planner_table, "max_pitch_acceleration_rad_s2"))
		{
			loaded.max_pitch_acceleration_rad_s2 = *v;
		}
		else
		{
			LOG_ERROR(kLogModule, "max_pitch_acceleration_rad_s2 must be a finite double");
			return false;
		}

		if(!read_vector2(*planner_table, "q_yaw", loaded.q_yaw))
		{
			LOG_ERROR(kLogModule, "q_yaw must be an array of 2 finite doubles");
			return false;
		}

		if(const auto v = read_double(*planner_table, "r_yaw"))
		{
			loaded.r_yaw = *v;
		}
		else
		{
			LOG_ERROR(kLogModule, "r_yaw must be a finite double");
			return false;
		}

		if(!read_vector2(*planner_table, "q_pitch", loaded.q_pitch))
		{
			LOG_ERROR(kLogModule, "q_pitch must be an array of 2 finite doubles");
			return false;
		}

		if(const auto v = read_double(*planner_table, "r_pitch"))
		{
			loaded.r_pitch = *v;
		}
		else
		{
			LOG_ERROR(kLogModule, "r_pitch must be a finite double");
			return false;
		}

		if(const auto v = read_double(*planner_table, "rho"))
		{
			loaded.rho = *v;
		}
		else
		{
			LOG_ERROR(kLogModule, "rho must be a finite double");
			return false;
		}

		if(const auto v = read_int(*planner_table, "max_iter"))
		{
			loaded.max_iter = static_cast<int>(*v);
		}
		else
		{
			LOG_ERROR(kLogModule, "max_iter must be an integer");
			return false;
		}

		try
		{
			validate_planner_config(loaded);
		}
		catch(const std::exception& exception)
		{
			LOG_ERROR(kLogModule, "planner configuration validation failed: {}", exception.what());
			return false;
		}

		config = std::move(loaded);
		return true;
	}

	bool load_planner_config(const std::string& config_path, PlannerConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			if(!load_planner_config_from_table(root, config))
			{
				LOG_ERROR(kLogModule, "planner configuration validation failed: {}", config_path);
				return false;
			}

			LOG_INFO(kLogModule, "loaded planner config: {}", config_path);
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
