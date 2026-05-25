/**
 * @file types.hpp
 * @brief 自瞄模块基础类型定义：装甲板颜色、类型、编号、优先级及其字符串转换。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_TYPES_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_TYPES_HPP

#include <cstdint>
#include <string_view> // 只读的<string>

namespace app::auto_aim
{

	/**
	 * @brief 装甲板颜色枚举。
	 */
	enum class ArmorColor : std::uint8_t
	{
		Red = 0,
		Blue,
		Extinguish,
		Purple,
		Unknown = 255
	};

	/**
	 * @brief 装甲板尺寸类型枚举。
	 */
	enum class ArmorType : std::uint8_t
	{
		Big = 0,
		Small,
		Unknown
	};

	/**
	 * @brief 装甲板编号（机器人 ID）枚举。
	 */
	enum class ArmorName : std::uint8_t
	{
		One = 0,
		Two,
		Three,
		Four,
		Five,
		Sentry,
		Outpost,
		Base,
		NotArmor = 255
	};

	/**
	 * @brief 打击优先级枚举。
	 */
	enum class ArmorPriority : std::uint8_t
	{
		First = 1,
		Second = 2,
		Third = 3,
		Fourth = 4,
		Fifth = 5,
		Unknown = 255
	};

	/**
	 * @brief 装甲板属性集合：颜色 + 编号 + 尺寸类型。
	 */
	struct ArmorProperty
	{
		ArmorColor color = ArmorColor::Unknown;
		ArmorName name = ArmorName::NotArmor;
		ArmorType type = ArmorType::Unknown;
	};

	/**
	 * @brief 将装甲板颜色枚举转换为可读字符串。
	 * @param color 装甲板颜色。
	 * @return 例如 "red"、"blue"、"unknown"。
	 */
	inline std::string_view to_string(ArmorColor color)
	{
		switch(color)
		{
		case ArmorColor::Red:
			return "red";
		case ArmorColor::Blue:
			return "blue";
		case ArmorColor::Extinguish:
			return "extinguish";
		case ArmorColor::Purple:
			return "purple";
		case ArmorColor::Unknown:
		default:
			return "unknown";
		}
	}

	/**
	 * @brief 将装甲板类型枚举转换为可读字符串。
	 * @param type 装甲板尺寸类型。
	 * @return 例如 "big"、"small"、"unknown"。
	 */
	inline std::string_view to_string(ArmorType type)
	{
		switch(type)
		{
		case ArmorType::Big:
			return "big";
		case ArmorType::Small:
			return "small";
		case ArmorType::Unknown:
		default:
			return "unknown";
		}
	}

	/**
	 * @brief 将装甲板编号枚举转换为可读字符串。
	 * @param name 装甲板编号。
	 * @return 例如 "one"、"two"、"sentry"、"base"、"not_armor"。
	 */
	inline std::string_view to_string(ArmorName name)
	{
		switch(name)
		{
		case ArmorName::One:
			return "one";
		case ArmorName::Two:
			return "two";
		case ArmorName::Three:
			return "three";
		case ArmorName::Four:
			return "four";
		case ArmorName::Five:
			return "five";
		case ArmorName::Sentry:
			return "sentry";
		case ArmorName::Outpost:
			return "outpost";
		case ArmorName::Base:
			return "base";
		case ArmorName::NotArmor:
		default:
			return "not_armor";
		}
	}

	/**
	 * @brief 将打击优先级枚举转换为可读字符串。
	 * @param priority 打击优先级。
	 * @return 例如 "first"、"second"、"unknown"。
	 */
	inline std::string_view to_string(ArmorPriority priority)
	{
		switch(priority)
		{
		case ArmorPriority::First:
			return "first";
		case ArmorPriority::Second:
			return "second";
		case ArmorPriority::Third:
			return "third";
		case ArmorPriority::Fourth:
			return "fourth";
		case ArmorPriority::Fifth:
			return "fifth";
		case ArmorPriority::Unknown:
		default:
			return "unknown";
		}
	}

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_TYPES_HPP
