#include "app/auto_aim/detector/detector_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "tools/logger.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "DETECTOR_CONFIG";

		std::string to_lower(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
				return static_cast<char>(std::tolower(character));
			});

			return value;
		}

		ArmorColor armor_color_from_string(std::string value)
		{
			value = to_lower(std::move(value));

			if(value == "red")
			{
				return ArmorColor::Red;
			}

			if(value == "blue")
			{
				return ArmorColor::Blue;
			}

			if(value == "extinguish")
			{
				return ArmorColor::Extinguish;
			}

			if(value == "purple")
			{
				return ArmorColor::Purple;
			}

			return ArmorColor::Unknown;
		}

		bool is_valid_probability(float value)
		{
			return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
		}

		bool validate_config(const DetectorConfig& config)
		{
			if(config.enemy_color == ArmorColor::Unknown)
			{
				LOG_ERROR(kLogModule, "enemy_color is invalid");

				return false;
			}

			if(!is_valid_probability(config.confidence_threshold))
			{
				LOG_ERROR(kLogModule,
				          "confidence_threshold must be "
				          "between 0 and 1");

				return false;
			}

			if(!is_valid_probability(config.nms_threshold))
			{
				LOG_ERROR(kLogModule,
				          "nms_threshold must be "
				          "between 0 and 1");

				return false;
			}

			if(!std::isfinite(config.min_armor_ratio) || config.min_armor_ratio <= 0.0)
			{
				LOG_ERROR(kLogModule, "min_armor_ratio must be positive");

				return false;
			}

			if(!std::isfinite(config.max_armor_ratio)
			   || config.max_armor_ratio < config.min_armor_ratio)
			{
				LOG_ERROR(kLogModule,
				          "max_armor_ratio must be greater "
				          "than or equal to min_armor_ratio");

				return false;
			}

			if(!std::isfinite(config.max_rectangular_error) || config.max_rectangular_error < 0.0)
			{
				LOG_ERROR(kLogModule,
				          "max_rectangular_error must be "
				          "non-negative");

				return false;
			}

			if(config.device.empty())
			{
				LOG_ERROR(kLogModule, "inference device cannot be empty");

				return false;
			}

			return true;
		}

		std::string resolve_model_path(const std::string& model_path, const std::string& config_path)
		{
			if(model_path.empty())
			{
				return {};
			}

			std::filesystem::path path(model_path);

			if(path.is_absolute())
			{
				return path.lexically_normal().string();
			}

#ifdef PROJECT_SOURCE_DIR
			path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
#else
			path = std::filesystem::path(config_path).parent_path() / path;
#endif

			return path.lexically_normal().string();
		}

	} // namespace

	bool load_detector_config(const std::string& config_path, DetectorConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			const auto* detector_table = root["detector"].as_table();

			if(detector_table == nullptr)
			{
				LOG_ERROR(kLogModule, "missing [detector] table in {}", config_path);

				return false;
			}

			DetectorConfig loaded_config;

			const std::string enemy_color =
			    (*detector_table)["enemy_color"].value_or(std::string("blue"));

			loaded_config.enemy_color = armor_color_from_string(enemy_color);

			loaded_config.confidence_threshold =
			    static_cast<float>((*detector_table)["confidence_threshold"].value_or(0.50));

			loaded_config.nms_threshold =
			    static_cast<float>((*detector_table)["nms_threshold"].value_or(0.45));

			loaded_config.min_armor_ratio = (*detector_table)["min_armor_ratio"].value_or(1.0);

			loaded_config.max_armor_ratio = (*detector_table)["max_armor_ratio"].value_or(6.0);

			loaded_config.max_rectangular_error =
			    (*detector_table)["max_rectangular_error"].value_or(0.50);

			loaded_config.enable_debug = (*detector_table)["enable_debug"].value_or(false);

			if(const auto* inference_table = root["inference"].as_table();
			   inference_table != nullptr)
			{
				const std::string model_path =
				    (*inference_table)["model_path"].value_or(std::string{});

				loaded_config.model_path = resolve_model_path(model_path, config_path);

				loaded_config.device = (*inference_table)["device"].value_or(std::string("CPU"));
			}

			if(!validate_config(loaded_config))
			{
				LOG_ERROR(kLogModule,
				          "detector configuration "
				          "validation failed: {}",
				          config_path);

				return false;
			}

			if(!loaded_config.model_path.empty())
			{
				std::error_code error_code;

				const bool model_exists =
				    std::filesystem::exists(loaded_config.model_path, error_code);

				if(error_code || !model_exists)
				{
					LOG_WARN(kLogModule,
					         "model file does not currently "
					         "exist: {}",
					         loaded_config.model_path);
				}
			}

			config = std::move(loaded_config);

			LOG_INFO(kLogModule,
			         "loaded detector config: "
			         "enemy={}, confidence={:.2f}, "
			         "nms={:.2f}, device={}",
			         to_string(config.enemy_color), config.confidence_threshold,
			         config.nms_threshold, config.device);

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
			LOG_ERROR(kLogModule,
			          "failed to load {}: "
			          "unknown exception",
			          config_path);

			return false;
		}
	}

} // namespace app::auto_aim
