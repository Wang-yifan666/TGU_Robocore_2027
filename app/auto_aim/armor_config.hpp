// 输入：
// toml::table 形式的 class_id_map（key 为 0-34，value 为 {color, name, type}）

// 输出：
// std::vector<ArmorProperty>，按下标（class_id）

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP

#include <vector>

#include "app/auto_aim/types.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	// 从 toml class_id_map 表加载 ArmorProperty vector
	// 返回的 vector 按下标（class_id）对齐
	std::vector<ArmorProperty> load_armor_properties(const toml::table& class_id_map);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_CONFIG_HPP
