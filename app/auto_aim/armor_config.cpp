#include "app/auto_aim/armor_config.hpp"

#include <map>
#include <string>

namespace app::auto_aim
{
	namespace
	{

		ArmorColor armor_color_from_string(std::string_view str)
		{
			if(str == "red")
			{
				return ArmorColor::Red;
			}
			if(str == "blue")
			{
				return ArmorColor::Blue;
			}
			if(str == "extinguish")
			{
				return ArmorColor::Extinguish;
			}
			if(str == "purple")
			{
				return ArmorColor::Purple;
			}
			return ArmorColor::Unknown;
		}

		ArmorType armor_type_from_string(std::string_view str)
		{
			if(str == "big")
			{
				return ArmorType::Big;
			}
			if(str == "small")
			{
				return ArmorType::Small;
			}
			return ArmorType::Unknown;
		}

		ArmorName armor_name_from_string(std::string_view str)
		{
			if(str == "one")
			{
				return ArmorName::One;
			}
			if(str == "two")
			{
				return ArmorName::Two;
			}
			if(str == "three")
			{
				return ArmorName::Three;
			}
			if(str == "four")
			{
				return ArmorName::Four;
			}
			if(str == "five")
			{
				return ArmorName::Five;
			}
			if(str == "sentry")
			{
				return ArmorName::Sentry;
			}
			if(str == "outpost")
			{
				return ArmorName::Outpost;
			}
			if(str == "base")
			{
				return ArmorName::Base;
			}
			return ArmorName::NotArmor;
		}

		ArmorProperty parse_armor_entry(const toml::table& entry)
		{
			ArmorProperty property;

			if(const auto color = entry["color"].value<std::string>())
			{
				property.color = armor_color_from_string(*color);
			}

			if(const auto name = entry["name"].value<std::string>())
			{
				property.name = armor_name_from_string(*name);
			}

			if(const auto type = entry["type"].value<std::string>())
			{
				property.type = armor_type_from_string(*type);
			}

			return property;
		}

	} // namespace

	std::vector<ArmorProperty> load_armor_properties(const toml::table& class_id_map)
	{
		// toml::table 的迭代顺序是字母序（0, 1, 10, 11, ...），
		// 所以需要用 map 按数字 key 排序后再构造 vector
		std::map<int, ArmorProperty> sorted;

		for(const auto& [key, value]: class_id_map)
		{
			const auto* entry = value.as_table();
			if(entry == nullptr)
			{
				continue;
			}

			const int id = std::stoi(std::string(key.str()));
			sorted[id] = parse_armor_entry(*entry);
		}

		std::vector<ArmorProperty> result;
		result.reserve(sorted.size());

		// 已排序，遍历填充
		// 注意：如果 class_id_map 有空洞（如缺少 key 5），结果 vector 对应位置会是默认构造值
		for(auto& [id, property]: sorted)
		{
			// 确保下标对齐：如果有空洞，先用默认值填充
			while(static_cast<int>(result.size()) < id)
			{
				result.emplace_back();
			}
			result.push_back(std::move(property));
		}

		return result;
	}

} // namespace app::auto_aim
