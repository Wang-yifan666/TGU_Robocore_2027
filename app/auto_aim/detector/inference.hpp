/**
 * @file inference.hpp
 * @brief 推理接口（Inference 抽象基类）与原始检测结果（RawDetection）的数据结构定义。
 *
 *   处理：
 * - RawDetection 数据结构定义（关键点、颜色/编号 ID、置信度）
 * - Inference 抽象基类声明（纯虚函数 infer / is_ready）
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_INFERENCE_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_INFERENCE_HPP

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace app::auto_aim
{

	struct RawDetection
	{
		std::array<cv::Point2f, 4> keypoints{};

		int color_id = -1;
		int number_id = -1;

		float confidence = 0.0F;
	};

	class Inference
	{
	public:
		virtual ~Inference() = default;

		virtual bool is_ready() const noexcept = 0;

		virtual std::vector<RawDetection> infer(const cv::Mat& image) = 0;
	};

} // namespace app::auto_aim

#endif
