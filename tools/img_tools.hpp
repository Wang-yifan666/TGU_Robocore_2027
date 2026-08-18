/**
 * @file img_tools.hpp
 * @brief 通用 OpenCV 图像绘制工具（业务无关）。
 *
 * 仅提供底层绘图原语：点、闭合轮廓、文本。
 * 不包含任何 app::auto_aim 等业务类型，禁止 tools -> app 反向依赖。
 */

#ifndef TGU_ROBOCORE_2027_TOOLS_IMG_TOOLS_HPP
#define TGU_ROBOCORE_2027_TOOLS_IMG_TOOLS_HPP

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace tools
{

	/**
	 * @brief 在图像上绘制实心圆点。
	 * @param image 目标图像（就地修改）。
	 * @param point 圆心（像素坐标）。
	 * @param color BGR 颜色。
	 * @param radius 圆半径（像素）。
	 */
	void draw_point(cv::Mat& image, const cv::Point& point, const cv::Scalar& color = {0, 0, 255},
	                int radius = 3);

	/**
	 * @brief 按闭合 contour 语义绘制点集（drawContours 自动连接末点至首点）。
	 * @param image 目标图像（就地修改）。
	 * @param points 轮廓点（整数坐标）。
	 * @param color BGR 颜色。
	 * @param thickness 线宽；负值表示填充。
	 */
	void draw_points(cv::Mat& image, const std::vector<cv::Point>& points,
	                 const cv::Scalar& color = {0, 0, 255}, int thickness = 2);

	/**
	 * @brief 按闭合 contour 语义绘制浮点坐标点集。
	 * @param image 目标图像（就地修改）。
	 * @param points 轮廓点（浮点坐标，内部转换为整数）。
	 * @param color BGR 颜色。
	 * @param thickness 线宽；负值表示填充。
	 */
	void draw_points(cv::Mat& image, const std::vector<cv::Point2f>& points,
	                 const cv::Scalar& color = {0, 0, 255}, int thickness = 2);

	/**
	 * @brief 在图像上绘制文本。
	 * @param image 目标图像（就地修改）。
	 * @param text 文本内容（ASCII）。
	 * @param origin 文本左下角锚点（像素坐标）。
	 * @param color BGR 颜色。
	 * @param font_scale 字号缩放。
	 * @param thickness 线宽。
	 */
	void draw_text(cv::Mat& image, const std::string& text, const cv::Point& origin,
	               const cv::Scalar& color = {0, 255, 255}, double font_scale = 1.0,
	               int thickness = 2);

} // namespace tools

#endif // TGU_ROBOCORE_2027_TOOLS_IMG_TOOLS_HPP