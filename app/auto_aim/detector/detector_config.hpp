/**
 * @file detector_config.hpp
 * @brief 检测器配置（DetectorConfig）的数据结构定义与 TOML 加载函数声明。
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_CONFIG_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_CONFIG_HPP

#include <string>

#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	struct DetectorConfig
	{
		ArmorColor enemy_color = ArmorColor::Blue;

		float confidence_threshold = 0.5F;
		float nms_threshold = 0.45F;

		double min_armor_ratio = 1.0;
		double max_armor_ratio = 6.0;
		double max_rectangular_error = 0.5;

		bool enable_debug = false;

		std::string model_path;
		std::string device = "CPU";
	};

	bool load_detector_config(const std::string& config_path, DetectorConfig& config);

} // namespace app::auto_aim

#endif
