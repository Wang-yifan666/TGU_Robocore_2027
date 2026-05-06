#include "app/auto_aim/armor.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#include "tools/tomlpp.hpp"

namespace
{

	// 将表拆到配置中
	// 从 toml 中解析单个 class_id 映射条目
	// 成功则返回 {color, name, type}，失败则返回默认值 {extinguish, not_armor, small}
	auto parse_armor_entry(const toml::table& entry) -> std::tuple<auto_aim::Color,
	                                                              auto_aim::ArmorName,
	                                                              auto_aim::ArmorType>
	{
		using auto_aim::ArmorName;
		using auto_aim::ArmorType;
		using auto_aim::Color;

		// 解析 color
		auto color = Color::extinguish;
		auto color_str = entry["color"].value<std::string>();

		if(color_str)
		{
			if(*color_str == "red")
				color = Color::red;
			else if(*color_str == "blue")
				color = Color::blue;
			else if(*color_str == "purple")
				color = Color::purple;
			// 其他情况默认 extinguish
		}

		// 解析 name
		auto name = ArmorName::not_armor;
		auto name_str = entry["name"].value<std::string>();
		
		if(name_str)
		{
			if(*name_str == "one")
				name = ArmorName::one;
			else if(*name_str == "two")
				name = ArmorName::two;
			else if(*name_str == "three")
				name = ArmorName::three;
			else if(*name_str == "four")
				name = ArmorName::four;
			else if(*name_str == "five")
				name = ArmorName::five;
			else if(*name_str == "sentry")
				name = ArmorName::sentry;
			else if(*name_str == "outpost")
				name = ArmorName::outpost;
			else if(*name_str == "base")
				name = ArmorName::base;
			// 其他情况默认 not_armor
		}

		// 解析 type
		auto type = ArmorType::small;
		auto type_str = entry["type"].value<std::string>();
		if(type_str && *type_str == "big")
		{
			type = ArmorType::big;
		}

		return {color, name, type};
	}

	// 默认从配置文件加载 armor_properties
	auto load_default_armor_properties() -> std::vector<std::tuple<auto_aim::Color,
	                                                               auto_aim::ArmorName,
	                                                               auto_aim::ArmorType>>
	{
		static auto config = toml::parse_file(
		    std::string(PROJECT_SOURCE_DIR) + "/config/app/auto_aim/armor_config.toml");
		auto class_id_map = config["armor"]["class_id_map"].as_table();
		if(class_id_map)
		{
			return auto_aim::load_armor_properties(*class_id_map);
		}
		return {};
	}

} // namespace

// armor_properties 的定义：编译时从配置文件加载
const std::vector<std::tuple<auto_aim::Color, auto_aim::ArmorName, auto_aim::ArmorType>>
    auto_aim::armor_properties = load_default_armor_properties();

namespace auto_aim
{

	namespace
	{

		void apply_offset(std::vector<cv::Point2f>& points, const cv::Point2f& offset)
		{
			for(auto& point: points)
			{
				point += offset;
			}
		}

		void calculate_geometry(Armor& armor, const std::vector<cv::Point2f>& armor_keypoints)
		{
			armor.center =
			    (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3])
			    / 4.0F;

			const auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
			const auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
			const auto max_width = std::max(left_width, right_width);

			const auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
			const auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
			const auto max_length = std::max(top_length, bottom_length);

			const auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2.0F;
			const auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2.0F;
			const auto left2right = right_center - left_center;

			const auto roll = std::atan2(left2right.y, left2right.x);

			const auto left_rectangular_error =
			    std::abs(std::atan2((armor_keypoints[3] - armor_keypoints[0]).y,
			                        (armor_keypoints[3] - armor_keypoints[0]).x)
			             - roll - CV_PI / 2.0);

			const auto right_rectangular_error =
			    std::abs(std::atan2((armor_keypoints[2] - armor_keypoints[1]).y,
			                        (armor_keypoints[2] - armor_keypoints[1]).x)
			             - roll - CV_PI / 2.0);

			// 防止除以 0
			armor.rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
			armor.ratio = max_width > 1e-6 ? max_length / max_width : 0.0;
		}

		void set_armor_property_by_class_id(Armor& armor, int class_id)
		{
			if(class_id >= 0 && class_id < static_cast<int>(armor_properties.size()))
			{
				const auto [color, name, type] = armor_properties[class_id];
				armor.color = color;
				armor.name = name;
				armor.type = type;
			}
			else
			{
				armor.color = blue;
				armor.name = not_armor;
				armor.type = small;
			}
		}

		void set_armor_property_by_yolo_id(Armor& armor, int color_id, int num_id)
		{
			armor.color = color_id == 0 ? blue : color_id == 1 ? red : extinguish;
			armor.name = num_id == 0 ? sentry
			    : num_id > 5         ? ArmorName(num_id)
			                         : ArmorName(num_id - 1);
			armor.type = num_id == 1 ? big : small;
		}

	} // namespace

	// load_armor_properties 的实现
	std::vector<std::tuple<Color, ArmorName, ArmorType>> load_armor_properties(
	    const toml::table& class_id_map)
	{
		// toml::table 的迭代顺序是字母序（0, 1, 10, 11, ...），
		// 所以需要用 map 按数字 key 排序后再构造 vector
		std::map<int, std::tuple<Color, ArmorName, ArmorType>> sorted;

		for(const auto& [key, value]: class_id_map)
		{
			auto entry = value.as_table();
			if(!entry)
			{
				continue;
			}
			int id = std::stoi(std::string(key.str()));
			sorted[id] = parse_armor_entry(*entry);
		}

		std::vector<std::tuple<Color, ArmorName, ArmorType>> result;
		result.reserve(sorted.size());
		for(auto& [id, entry]: sorted)
		{
			result.push_back(std::move(entry));
		}
		return result;
	}

	Lightbar::Lightbar(const cv::RotatedRect& rotated_rect, std::size_t id):

	id(id), rotated_rect(rotated_rect)
	{
		std::vector<cv::Point2f> corners(4);
		rotated_rect.points(corners.data());

		std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
			return a.y < b.y;
		});

		center = rotated_rect.center;
		top = (corners[0] + corners[1]) / 2.0F;
		bottom = (corners[2] + corners[3]) / 2.0F;
		top2bottom = bottom - top;

		points.emplace_back(top);
		points.emplace_back(bottom);

		width = cv::norm(corners[0] - corners[1]);
		length = cv::norm(top2bottom);
		angle = std::atan2(top2bottom.y, top2bottom.x);
		angle_error = std::abs(angle - CV_PI / 2.0);
		ratio = width > 1e-6 ? length / width : 0.0;
	}

	Armor::Armor(const Lightbar& left, const Lightbar& right):

	color(left.color), left(left), right(right), duplicated(false)
	{
		center = (left.center + right.center) / 2.0F;

		points.emplace_back(left.top);
		points.emplace_back(right.top);
		points.emplace_back(right.bottom);
		points.emplace_back(left.bottom);

		const auto left2right = right.center - left.center;
		const auto width = cv::norm(left2right);

		const auto max_lightbar_length = std::max(left.length, right.length);
		const auto min_lightbar_length = std::min(left.length, right.length);

		ratio = max_lightbar_length > 1e-6 ? width / max_lightbar_length : 0.0;
		side_ratio = min_lightbar_length > 1e-6 ? max_lightbar_length / min_lightbar_length : 0.0;

		const auto roll = std::atan2(left2right.y, left2right.x);
		const auto left_rectangular_error = std::abs(left.angle - roll - CV_PI / 2.0);
		const auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2.0);

		rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
	}

	Armor::Armor(int class_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints):

	class_id(class_id), box(box), confidence(confidence), points(armor_keypoints)
	{
		calculate_geometry(*this, armor_keypoints);
		set_armor_property_by_class_id(*this, class_id);
	}

	Armor::Armor(int class_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset):

	class_id(class_id), box(box), confidence(confidence)
	{
		apply_offset(armor_keypoints, offset);
		points = armor_keypoints;

		calculate_geometry(*this, armor_keypoints);
		set_armor_property_by_class_id(*this, class_id);
	}

	Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints):

	box(box), confidence(confidence), points(armor_keypoints)
	{
		calculate_geometry(*this, armor_keypoints);
		set_armor_property_by_yolo_id(*this, color_id, num_id);
	}

	Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset):

	box(box), confidence(confidence)
	{
		apply_offset(armor_keypoints, offset);
		points = armor_keypoints;

		calculate_geometry(*this, armor_keypoints);
		set_armor_property_by_yolo_id(*this, color_id, num_id);
	}

} // namespace auto_aim
