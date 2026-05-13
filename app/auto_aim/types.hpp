// types.hpp

// 包含如下：
// 装甲板颜色（Red / Blue / Extinguish / Purple / Unknown）
// 装甲板类型（Big / Small / Unknown）
// 装甲板编号
// 打击优先级
// 上述三个属性的集合（color + name + type）
// 每个枚举类型对应的字符串转换函数

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_TYPES_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_TYPES_HPP

#include <cstdint>
#include <string_view> // 只读的<string>

namespace app::auto_aim
{

	enum class ArmorColor : std::uint8_t
	{
		Red = 0,
		Blue,
		Extinguish,
		Purple,
		Unknown = 255
	};

	enum class ArmorType : std::uint8_t
	{
		Big = 0,
		Small,
		Unknown
	};

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

	enum class ArmorPriority : std::uint8_t
	{
		First = 1,
		Second = 2,
		Third = 3,
		Fourth = 4,
		Fifth = 5,
		Unknown = 255
	};

	struct ArmorProperty
	{
		ArmorColor color = ArmorColor::Unknown;
		ArmorName name = ArmorName::NotArmor;
		ArmorType type = ArmorType::Unknown;
	};

	std::string_view to_string(ArmorColor color)
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

	std::string_view to_string(ArmorType type)
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

	std::string_view to_string(ArmorName name)
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

	std::string_view to_string(ArmorPriority priority)
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
