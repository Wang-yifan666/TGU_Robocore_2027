#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "tools/tomlpp.hpp"

namespace auto_aim
{

	enum Color
	{
		red,
		blue,
		extinguish,
		purple
	};

	const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

	enum ArmorType
	{
		big,
		small
	};

	const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

	enum ArmorName
	{
		one,
		two,
		three,
		four,
		five,
		sentry,
		outpost,
		base,
		not_armor
	};

	const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
	                                              "sentry", "outpost", "base",  "not_armor"};

	enum ArmorPriority
	{
		first = 1,
		second,
		third,
		forth,
		fifth
	};

	// armor_properties 表，从 config/app/auto_aim/armor_config.toml 加载
	// 映射: class_id → {color, name, type}
	extern const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties;

	// 从 toml 配置加载 armor_properties 表
	// 调用者需传入已解析好的 config["armor"]["class_id_map"] 子表
	// 返回解析好的 vector
	std::vector<std::tuple<Color, ArmorName, ArmorType>> load_armor_properties(
	    const toml::table& class_id_map);

	struct Lightbar
	{
		std::size_t id = 0;
		Color color = extinguish;

		cv::Point2f center;
		cv::Point2f top;
		cv::Point2f bottom;
		cv::Point2f top2bottom;
		std::vector<cv::Point2f> points;

		double angle = 0.0;
		double angle_error = 0.0;
		double length = 0.0;
		double width = 0.0;
		double ratio = 0.0;

		cv::RotatedRect rotated_rect;

		Lightbar() = default;
		Lightbar(const cv::RotatedRect& rotated_rect, std::size_t id);
	};

	struct Armor
	{
		Color color = extinguish;
		Lightbar left;
		Lightbar right;

		// 是左右灯条中心的平均值，不是对角线交点，不能作为实际中心
		cv::Point2f center;
		cv::Point2f center_norm; // 归一化坐标

		std::vector<cv::Point2f> points;

		double ratio = 0.0;       // 两灯条的中点连线与长灯条的长度之比
		double side_ratio;        // 长灯条与短灯条的长度之比
		double rectangular_error; // 灯条和中点连线所成夹角与π/2的差值

		ArmorType type = small;
		ArmorName name = not_armor;
		ArmorPriority priority = fifth;

		int class_id = -1;
		cv::Rect box;
		cv::Mat pattern;
		double confidence = 0.0;
		bool duplicated = false;

		// 全部初始化为 0 向量
		Eigen::Vector3d xyz_in_gimbal = Eigen::Vector3d::Zero();
		Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();

		Eigen::Vector3d ypr_in_gimbal = Eigen::Vector3d::Zero();
		Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
		Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();

		double yaw_raw = 0.0;

		// 构造函数
		Armor() = default;
		Armor(const Lightbar& left, const Lightbar& right);

		Armor(int class_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints);

		Armor(int class_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);

		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints);

		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
	};

} // namespace auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
