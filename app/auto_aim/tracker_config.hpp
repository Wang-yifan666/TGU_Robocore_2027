/**
 * @file tracker_config.hpp
 * @brief Tracker 配置加载：TOML → TrackerConfig。
 *
 * 职责：
 * - 校验字段存在、数值 finite、尺寸（P0/R）合法；
 * - 加载失败返回 false，不静默 fallback。
 */

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_TRACKER_CONFIG_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_TRACKER_CONFIG_HPP

#include <string>

#include "app/auto_aim/tracker.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 从已解析的 toml::table 加载 TrackerConfig（纯逻辑，便于单测）。
	 */
	bool load_tracker_config_from_table(const toml::table& root, TrackerConfig& config);

	/**
	 * @brief 从 TOML 文件加载 TrackerConfig。
	 */
	bool load_tracker_config(const std::string& config_path, TrackerConfig& config);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_TRACKER_CONFIG_HPP