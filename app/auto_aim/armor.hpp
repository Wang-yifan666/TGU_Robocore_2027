/**
 * @file armor.hpp
 * @brief 灯条（Lightbar）与装甲板（Armor）的数据结构定义与几何计算。
 *
 * 职责范围：
 * - Lightbar / Armor 数据结构定义
 * - 灯条几何参数计算（角度、长宽比等）
 * - 装甲板几何参数计算（中心点、矩形误差、宽高比等）
 * - 支持 class_id 映射与 YOLO 双 id 映射两种构造方式
 *
 * 不负责：
 * - TOML 解析（见 armor_config.hpp）
 * - PnP 求解（见 solver.hpp）
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "app/auto_aim/types.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 灯条数据结构，包含几何参数与旋转矩形信息。
	 */
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

		/**
		 * @brief 从 OpenCV RotatedRect 构造灯条，自动计算几何参数。
		 * @param rotated_rect 检测到的旋转矩形。
		 * @param id 灯条唯一标识。
		 */
		Lightbar(const cv::RotatedRect& rotated_rect, std::size_t id);
	};

	/**
	 * @brief 装甲板数据结构，包含几何参数、检测信息与空间坐标。
	 *
	 * 支持两种构造方式：
	 * - 灯条匹配构造（Lightbar 左右配对）
	 * - 检测结果构造（class_id 映射 或 YOLO 双 id 映射）
	 */
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

		Armor() = default;

		/**
		 * @brief 从左右灯条构造装甲板，自动计算几何参数（中心点、宽高比、矩形误差等）。
		 * @param left 左侧灯条。
		 * @param right 右侧灯条。
		 */
		Armor(const Lightbar& left, const Lightbar& right);

		/**
		 * @brief 从检测结果构造装甲板（class_id 映射方式）。
		 * @param class_id 类别 ID，映射到 ArmorProperty。
		 * @param confidence 检测置信度。
		 * @param box 检测框。
		 * @param armor_keypoints 装甲板四个角点，顺序为左上、右上、右下、左下。
		 * @param armor_properties 由外部传入的 ArmorProperty 数组，按下标对齐 class_id。
		 */
		Armor(int class_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints,
		      const std::vector<ArmorProperty>& armor_properties);

		/**
		 * @brief 从检测结果构造装甲板（class_id 映射方式），支持偏移修正。
		 * @param class_id 类别 ID。
		 * @param confidence 检测置信度。
		 * @param box 检测框。
		 * @param armor_keypoints 装甲板四个角点。
		 * @param offset 偏移量（例如图像裁剪补偿），应用到所有角点。
		 * @param armor_properties 由外部传入的 ArmorProperty 数组。
		 */
		Armor(int class_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset,
		      const std::vector<ArmorProperty>& armor_properties);

		/**
		 * @brief 从检测结果构造装甲板（YOLO 双 id 兼容方式）。
		 * @note 建议优先使用 class_id 映射的构造方式。
		 * @param color_id 颜色 ID（0=蓝, 1=红, 2=灭灯, 3=紫）。
		 * @param num_id 编号 ID（0-7 对应哨兵/1-5/前哨站/基地）。
		 * @param confidence 检测置信度。
		 * @param box 检测框。
		 * @param armor_keypoints 装甲板四个角点。
		 */
		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints);

		/**
		 * @brief 从检测结果构造装甲板（YOLO 双 id 兼容方式），支持偏移修正。
		 * @note 建议优先使用 class_id 映射的构造方式。
		 * @param color_id 颜色 ID。
		 * @param num_id 编号 ID。
		 * @param confidence 检测置信度。
		 * @param box 检测框。
		 * @param armor_keypoints 装甲板四个角点。
		 * @param offset 偏移量，应用到所有角点。
		 */
		Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
		      std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_ARMOR_HPP
