#include "tools/img_tools.hpp"

#include <vector>

#include <opencv2/imgproc.hpp>

namespace tools
{

	void draw_point(cv::Mat& image, const cv::Point& point, const cv::Scalar& color, int radius)
	{
		cv::circle(image, point, radius, color, -1);
	}

	void draw_points(cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color,
	                 int thickness)
	{
		std::vector<std::vector<cv::Point>> contours = {points};
		cv::drawContours(image, contours, -1, color, thickness);
	}

	void draw_points(cv::Mat& image, const std::vector<cv::Point2f>& points,
	                 const cv::Scalar& color, int thickness)
	{
		std::vector<cv::Point> int_points(points.begin(), points.end());
		draw_points(image, int_points, color, thickness);
	}

	void draw_text(cv::Mat& image, const std::string& text, const cv::Point& origin,
	               const cv::Scalar& color, double font_scale, int thickness)
	{
		cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
	}

} // namespace tools