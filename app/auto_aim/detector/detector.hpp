/**
 * @file detector.hpp
 * @brief 检测器主类（Detector）与检测结果（DetectionResult）的定义。
 *
 *   处理：
 * - DetectionResult 数据结构定义（包含装甲板列表与耗时统计）
 * - Detector 类的声明（推理 + 后处理 + NMS 管线）
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "app/auto_aim/armor.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/inference.hpp"

namespace app::auto_aim
{

	struct DetectionResult
	{
		std::vector<Armor> armors;

		double inference_time_ms = 0.0;
		double postprocess_time_ms = 0.0;

		std::uint64_t frame_id = 0;
	};

	class Detector
	{
	public:
		Detector(DetectorConfig config, std::unique_ptr<Inference> inference);

		bool is_ready() const noexcept;

		DetectionResult detect(const cv::Mat& image, std::uint64_t frame_id);

	private:
		bool is_valid_armor(const Armor& armor) const;

		Armor convert_detection(const RawDetection& detection) const;

	private:
		DetectorConfig config_;
		std::unique_ptr<Inference> inference_;
	};

} // namespace app::auto_aim

#endif
