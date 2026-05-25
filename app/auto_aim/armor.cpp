#include "app/auto_aim/armor.hpp"

#include <algorithm>
#include <cmath>

namespace app::auto_aim
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

		bool check_keypoints(const std::vector<cv::Point2f>& keypoints)
		{
			return keypoints.size() == 4;
		}

		void set_invalid_armor(Armor& armor)
		{
			armor.color = ArmorColor::Unknown;
			armor.name = ArmorName::NotArmor;
			armor.type = ArmorType::Unknown;
			armor.priority = ArmorPriority::Unknown;
			armor.duplicated = false;
		}

		bool calculate_geometry(Armor& armor, const std::vector<cv::Point2f>& armor_keypoints)
		{
			if(!check_keypoints(armor_keypoints))
			{
				set_invalid_armor(armor);
				return false;
			}

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

			return true;
		}

		void set_armor_property_by_class_id(Armor& armor, int class_id,
		                                    const std::vector<ArmorProperty>& armor_properties)
		{
			if(class_id >= 0 && class_id < static_cast<int>(armor_properties.size()))
			{
				const auto& property = armor_properties[static_cast<std::size_t>(class_id)];

				armor.color = property.color;
				armor.name = property.name;
				armor.type = property.type;
			}
			else
			{
				set_invalid_armor(armor);
			}
		}

		// ---- YOLO 双 id 映射函数 ----

		ArmorColor color_from_yolo_id(int color_id)
		{
			switch(color_id)
			{
			case 0:
				return ArmorColor::Blue;
			case 1:
				return ArmorColor::Red;
			case 2:
				return ArmorColor::Extinguish;
			case 3:
				return ArmorColor::Purple;
			default:
				return ArmorColor::Unknown;
			}
		}

		ArmorName name_from_yolo_id(int num_id)
		{
			switch(num_id)
			{
			case 0:
				return ArmorName::Sentry;
			case 1:
				return ArmorName::One;
			case 2:
				return ArmorName::Two;
			case 3:
				return ArmorName::Three;
			case 4:
				return ArmorName::Four;
			case 5:
				return ArmorName::Five;
			case 6:
				return ArmorName::Outpost;
			case 7:
				return ArmorName::Base;
			default:
				return ArmorName::NotArmor;
			}
		}

		ArmorType type_from_yolo_id(int num_id)
		{
			// class_id 映射优先；此处按配置惯例：id=1 为 big，其余 small
			if(num_id == 1)
			{
				return ArmorType::Big;
			}

			if(num_id >= 0 && num_id <= 7)
			{
				return ArmorType::Small;
			}

			return ArmorType::Unknown;
		}

		void set_armor_property_by_yolo_id(Armor& armor, int color_id, int num_id)
		{
			armor.color = color_from_yolo_id(color_id);
			armor.name = name_from_yolo_id(num_id);
			armor.type = type_from_yolo_id(num_id);

			if(armor.color == ArmorColor::Unknown || armor.name == ArmorName::NotArmor)
			{
				armor.type = ArmorType::Unknown;
				armor.priority = ArmorPriority::Unknown;
			}
		}

	} // namespace

	// Lightbar 实现

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

	// Armor 实现

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
	             std::vector<cv::Point2f> armor_keypoints,
	             const std::vector<ArmorProperty>& armor_properties):
	class_id(class_id), box(box), confidence(confidence), points(std::move(armor_keypoints))
	{
		if(calculate_geometry(*this, points))
		{
			set_armor_property_by_class_id(*this, class_id, armor_properties);
		}
	}

	Armor::Armor(int class_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset,
	             const std::vector<ArmorProperty>& armor_properties):
	class_id(class_id), box(box), confidence(confidence)
	{
		apply_offset(armor_keypoints, offset);
		points = std::move(armor_keypoints);

		if(calculate_geometry(*this, points))
		{
			set_armor_property_by_class_id(*this, class_id, armor_properties);
		}
	}

	Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints):
	box(box), confidence(confidence), points(std::move(armor_keypoints))
	{
		calculate_geometry(*this, points);
		set_armor_property_by_yolo_id(*this, color_id, num_id);
	}

	Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
	             std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset):
	box(box), confidence(confidence)
	{
		apply_offset(armor_keypoints, offset);
		points = std::move(armor_keypoints);

		calculate_geometry(*this, points);
		set_armor_property_by_yolo_id(*this, color_id, num_id);
	}

} // namespace app::auto_aim
