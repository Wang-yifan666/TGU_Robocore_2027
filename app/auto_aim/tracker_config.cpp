/**
 * @file tracker_config.cpp
 * @brief Tracker 配置加载实现。
 */

#include "app/auto_aim/tracker_config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tools/logger.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "TRACKER_CONFIG";

		std::optional<std::vector<double>> read_double_array(const toml::array* array)
		{
			if(array == nullptr)
			{
				return std::nullopt;
			}

			std::vector<double> values;
			values.reserve(array->size());

			for(const auto& element: *array)
			{
				const auto value = element.value<double>();

				if(!value)
				{
					return std::nullopt;
				}

				values.push_back(*value);
			}

			return values;
		}

		std::optional<int> read_int(const toml::table& table, std::string_view key)
		{
			const auto value = table[key].value<std::int64_t>();

			if(!value)
			{
				return std::nullopt;
			}

			const std::int64_t v = *value;
			if(v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
			{
				return std::nullopt;
			}

			return static_cast<int>(v);
		}

		std::optional<double> read_double(const toml::table& table, std::string_view key)
		{
			return table[key].value<double>();
		}

		bool read_diagonal_covariance(const toml::table& root, std::string_view key, int dim,
		                              Eigen::MatrixXd& out)
		{
			const auto* arr = root[key].as_array();

			if(arr == nullptr)
			{
				LOG_ERROR(kLogModule, "{} must be an array", key);
				return false;
			}

			const auto values = read_double_array(arr);

			if(!values)
			{
				LOG_ERROR(kLogModule, "{} must contain numeric values", key);
				return false;
			}

			if(static_cast<int>(values->size()) != dim
			   || !std::all_of(values->begin(), values->end(),
			                   [](double v) { return std::isfinite(v); }))
			{
				LOG_ERROR(kLogModule, "{} must contain exactly {} finite values", key, dim);
				return false;
			}

			out = Eigen::MatrixXd::Zero(dim, dim);

			for(int i = 0; i < dim; ++i)
			{
				out(i, i) = (*values)[static_cast<std::size_t>(i)];
			}

			return true;
		}

	} // namespace

	bool load_tracker_config_from_table(const toml::table& root, TrackerConfig& config)
	{
		const auto* lifecycle = root["lifecycle"].as_table();
		const auto* association = root["association"].as_table();
		const auto* process_noise = root["process_noise"].as_table();
		const auto* radius_table = root["radius"].as_table();
		const auto* radius_profile = root["radius_profile"].as_table();

		if(lifecycle == nullptr || association == nullptr || process_noise == nullptr
		   || radius_table == nullptr || radius_profile == nullptr)
		{
			LOG_ERROR(kLogModule,
			          "missing required table: lifecycle/association/process_noise/radius/"
			          "radius_profile");
			return false;
		}

		TrackerConfig loaded;

		// lifecycle。
		auto confirm = read_int(*lifecycle, "detecting_confirm_hits");
		if(!confirm || *confirm < 1)
		{
			LOG_ERROR(kLogModule, "detecting_confirm_hits must be >= 1");
			return false;
		}

		auto detecting_misses = read_int(*lifecycle, "detecting_max_misses");
		if(!detecting_misses || *detecting_misses < 0)
		{
			LOG_ERROR(kLogModule, "detecting_max_misses must be >= 0");
			return false;
		}

		auto temp_misses = read_int(*lifecycle, "temp_lost_max_misses");
		if(!temp_misses || *temp_misses < 0)
		{
			LOG_ERROR(kLogModule, "temp_lost_max_misses must be >= 0");
			return false;
		}

		auto max_dt = read_double(*lifecycle, "max_dt_s");
		if(!max_dt || !std::isfinite(*max_dt) || *max_dt <= 0.0)
		{
			LOG_ERROR(kLogModule, "max_dt_s must be finite and > 0");
			return false;
		}

		loaded.detecting_confirm_hits = *confirm;
		loaded.detecting_max_misses = *detecting_misses;
		loaded.temp_lost_max_misses = *temp_misses;
		loaded.max_dt_s = *max_dt;

		// association。
		auto assoc = [&](std::string_view key, double& out, bool positive) {
			auto v = read_double(*association, key);

			if(!v || !std::isfinite(*v) || (positive ? (*v <= 0.0) : (*v < 0.0)))
			{
				LOG_ERROR(kLogModule, "{} is invalid", key);
				return false;
			}

			out = *v;
			return true;
		};

		if(!assoc("max_position_error_m", loaded.association.max_position_error_m, false)
		   || !assoc("max_yaw_error_rad", loaded.association.max_yaw_error_rad, false)
		   || !assoc("position_score_scale_m", loaded.association.position_score_scale_m, true)
		   || !assoc("yaw_score_scale_rad", loaded.association.yaw_score_scale_rad, true))
		{
			return false;
		}

		// initial covariance diag (11) 与 measurement covariance diag (4)。
		if(!read_diagonal_covariance(root, "initial_covariance_diag", kTargetStateDim,
		                             loaded.initial_covariance)
		   || !read_diagonal_covariance(root, "measurement_covariance_diag", kTargetMeasurementDim,
		                                loaded.measurement_covariance))
		{
			return false;
		}

		// process noise。
		auto pnoise = [&](std::string_view key, double& out) {
			auto v = read_double(*process_noise, key);

			if(!v || !std::isfinite(*v) || *v < 0.0)
			{
				LOG_ERROR(kLogModule, "{} must be finite and >= 0", key);
				return false;
			}

			out = *v;
			return true;
		};

		if(!pnoise("translation_accel_variance",
		           loaded.process_noise.translation_accel_variance)
		   || !pnoise("yaw_accel_variance", loaded.process_noise.yaw_accel_variance)
		   || !pnoise("radius_random_walk_variance",
		              loaded.process_noise.radius_random_walk_variance)
		   || !pnoise("delta_radius_random_walk_variance",
		              loaded.process_noise.delta_radius_random_walk_variance)
		   || !pnoise("delta_z_random_walk_variance",
		              loaded.process_noise.delta_z_random_walk_variance))
		{
			return false;
		}

		// radius bounds。
		auto min_r = read_double(*radius_table, "min_radius_m");
		auto max_r = read_double(*radius_table, "max_radius_m");

		if(!min_r || !max_r || !std::isfinite(*min_r) || !std::isfinite(*max_r)
		   || *min_r <= 0.0 || *max_r < *min_r)
		{
			LOG_ERROR(kLogModule, "radius bounds must satisfy 0 < min <= max");
			return false;
		}

		loaded.min_radius_m = *min_r;
		loaded.max_radius_m = *max_r;

		// radius profile（除 positive finite 外，还必须落在 [min_radius_m, max_radius_m]）。
		auto rp = [&](std::string_view key, double& out) {
			auto v = read_double(*radius_profile, key);

			if(!v || !std::isfinite(*v) || *v <= 0.0)
			{
				LOG_ERROR(kLogModule, "{} must be finite and > 0", key);
				return false;
			}

			if(*v < loaded.min_radius_m || *v > loaded.max_radius_m)
			{
				LOG_ERROR(kLogModule, "{} must lie within [min_radius_m, max_radius_m]", key);
				return false;
			}

			out = *v;
			return true;
		};

		if(!rp("balance_2", loaded.radius_profile.balance_2)
		   || !rp("outpost_3", loaded.radius_profile.outpost_3)
		   || !rp("base_3", loaded.radius_profile.base_3)
		   || !rp("default_4", loaded.radius_profile.default_4))
		{
			return false;
		}

		config = std::move(loaded);

		return true;
	}

	bool load_tracker_config(const std::string& config_path, TrackerConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			if(!load_tracker_config_from_table(root, config))
			{
				LOG_ERROR(kLogModule, "tracker configuration validation failed: {}", config_path);
				return false;
			}

			LOG_INFO(kLogModule, "loaded tracker config: {}", config_path);

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