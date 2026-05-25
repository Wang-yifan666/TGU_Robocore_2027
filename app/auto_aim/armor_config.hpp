/**
 * @file armor_config.hpp
 * @brief 从 TOML 配置加载装甲板属性映射表（class_id → ArmorProperty）。
 */

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP

#include <vector>

#include "app/auto_aim/types.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 从 toml::table 形式的 class_id_map 加载 ArmorProperty 数组。
	 * @param class_id_map toml 表，key 为数字（class_id），value 为 {color, name, type}。
	 * @return 按下标（class_id）对齐的 ArmorProperty vector。
	 * @note 如果 class_id_map 中存在空洞（如缺少 key 5），结果 vector 对应位置为默认构造值。
	 */
	std::vector<ArmorProperty> load_armor_properties(const toml::table& class_id_map);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP
