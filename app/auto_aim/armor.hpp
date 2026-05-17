// 输入：
// 1. 灯条检测结果 Lightbar
// 2. 检测结果 (class_id, confidence, box, armor_keypoints)
// 3. ArmorProperty vector（由外部传入，不从全局或 TOML 读取）

// 输出：
// 1. Armor 结构体（color, left/right, center, type, name, priority）
// 2. 空间坐标 (xyz_in_gimbal / xyz_in_world / ypr / ypd)
// 3. 归一化坐标 (center_norm)

// 职责：
// Armor / Lightbar 的数据结构定义与几何计算
// 不负责 TOML 解析（见 armor_config.hpp）
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	struct Lightbar
	{
		std::size_t id = 0;

		ArmorColor color = ArmorColor::Unknown;

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
		ArmorColor color = ArmorColor::Unknown;
		Lightbar left;
		Lightbar right;

		// 是左右灯条中心的平均值，不是对角线交点，不能作为实际中心
		cv::Point2f center;
		cv::Point2f center_norm; // 归一化坐标

		std::vector<cv::Point2f> points;

		double ratio = 0.0;             // 两灯条的中点连线与长灯条的长度之比
		double side_ratio = 0.0;        // 长灯条与短灯条的长度之比
		double rectangular_error = 0.0; // 灯条和中点连线所成夹角与π/2的差值

		ArmorType type = ArmorType::Unknown;
		ArmorName name = ArmorName::NotArmor;
		ArmorPriority priority = ArmorPriority::Unknown;

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
		      std::vector<cv::Point2f> armor_keypoints,
		      const std::vector<ArmorProperty>& armor_properties);

		Armor(int class_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset,
		      const std::vector<ArmorProperty>& armor_properties);

		// 以下两个是 YOLO 双 id（color_id + num_id）的兼容构造方式
		// 建议优先使用 class_id 映射
		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints);

		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
