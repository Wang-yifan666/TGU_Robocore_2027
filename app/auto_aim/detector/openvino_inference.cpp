#include "app/auto_aim/detector/openvino_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/logger.hpp"

namespace app::auto_aim::detector_detail
{
	namespace
	{
		constexpr std::string_view kLogModule = "OPENVINO_INFERENCE";

		float sigmoid(float x)
		{
			if(x >= 0.0F)
			{
				return 1.0F / (1.0F + std::exp(-x));
			}

			const float ex = std::exp(x);
			return ex / (1.0F + ex);
		}

		int argmax(const float* begin, std::size_t count)
		{
			int best = 0;

			for(std::size_t i = 1; i < count; ++i)
			{
				if(begin[i] > begin[static_cast<std::size_t>(best)])
				{
					best = static_cast<int>(i);
				}
			}

			return best;
		}
	} // namespace

	double letterbox_scale(const int src_h, const int src_w, const int dst_h, const int dst_w)
	{
		const auto x_scale = static_cast<double>(dst_h) / static_cast<double>(src_h);
		const auto y_scale = static_cast<double>(dst_w) / static_cast<double>(src_w);

		return std::min(x_scale, y_scale);
	}

	cv::Size letterbox_size(const int src_h, const int src_w, const double scale)
	{
		const auto h = static_cast<int>(static_cast<double>(src_h) * scale);
		const auto w = static_cast<int>(static_cast<double>(src_w) * scale);

		return {w, h};
	}

	cv::Point2f map_point_to_original(const cv::Point2f& point, const double scale)
	{
		return {static_cast<float>(static_cast<double>(point.x) / scale),
		        static_cast<float>(static_cast<double>(point.y) / scale)};
	}

	DecodedRow decode_row(const float* row, const double scale, const float confidence_threshold)
	{
		DecodedRow result;
		result.accepted = false;

		// 防御：任何非有限数值都视为非法行。
		for(std::size_t col = 0; col < kYoloV5RowWidth; ++col)
		{
			if(!std::isfinite(row[col]))
			{
				return result;
			}
		}

		// col[8] 为 objectness 原始 logit，需 sigmoid。
		const float confidence = sigmoid(row[8]);

		if(confidence < confidence_threshold)
		{
			return result;
		}

		// 关键点顺序固定复现旧代码的 {col0/1, col6/7, col4/5, col2/3}。
		// 映射到原图的 [左上, 右上, 右下, 左下]。
		const std::array<cv::Point2f, 4> keypoints = {
		    map_point_to_original(cv::Point2f(row[0], row[1]), scale),
		    map_point_to_original(cv::Point2f(row[6], row[7]), scale),
		    map_point_to_original(cv::Point2f(row[4], row[5]), scale),
		    map_point_to_original(cv::Point2f(row[2], row[3]), scale),
		};

		// 颜色 one-hot：col[9..12]（4 类，0..3）。
		const int color_id = argmax(row + 9, 4);

		// 编号 one-hot：col[13..21]（9 类，0..8）。
		const int number_id = argmax(row + 13, 9);

		result.detection.keypoints = keypoints;
		result.detection.color_id = color_id;
		result.detection.number_id = number_id;
		result.detection.confidence = confidence;
		result.accepted = true;

		return result;
	}

} // namespace app::auto_aim::detector_detail

namespace app::auto_aim
{
	namespace
	{
		constexpr std::string_view kLogModule = "OPENVINO_INFERENCE";
	}

	OpenVINOInference::OpenVINOInference(std::string model_path, std::string device,
	                                     const float confidence_threshold):
	model_path_(std::move(model_path)),
	device_(std::move(device)), confidence_threshold_(confidence_threshold)
	{
		try
		{
			auto model = core_.read_model(model_path_);

			// 从模型本身读取输入 H/W，不硬编码尺寸。
			const ov::PartialShape input_shape = model->input().get_partial_shape();

			if(!input_shape.rank().is_static() || input_shape.rank().get_length() != 4)
			{
				throw std::runtime_error(std::format(
				    "unexpected input rank: {} (expected 4)",
				    input_shape.rank().is_static()
				        ? std::to_string(input_shape.rank().get_length())
				        : std::string("dynamic")));
			}

			const auto height = input_shape[2];
			const auto width = input_shape[3];

			if(!height.is_static() || !width.is_static())
			{
				throw std::runtime_error("input height/width is dynamic, cannot infer letterbox size");
			}

			input_h_ = static_cast<std::size_t>(height.get_length());
			input_w_ = static_cast<std::size_t>(width.get_length());

			// 与 sp_vision_25 一致的预处理：u8 NHWC BGR -> f32 RGB / 255 -> NCHW。
			ov::preprocess::PrePostProcessor ppp(model);
			auto& input = ppp.input();

			input.tensor()
			    .set_element_type(ov::element::u8)
			    .set_shape({1, static_cast<ov::Dimension::value_type>(input_h_),
			                static_cast<ov::Dimension::value_type>(input_w_), 3})
			    .set_layout("NHWC")
			    .set_color_format(ov::preprocess::ColorFormat::BGR);

			input.model().set_layout("NCHW");

			input.preprocess()
			    .convert_element_type(ov::element::f32)
			    .convert_color(ov::preprocess::ColorFormat::RGB)
			    .scale(255.0);

			model = ppp.build();

			compiled_model_ = core_.compile_model(
			    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

			ready_ = true;

			LOG_INFO(kLogModule, "OpenVINO model loaded: {} (device={}, input={}x{})", model_path_,
			         device_, input_h_, input_w_);
		}
		catch(const std::exception& exception)
		{
			ready_ = false;
			LOG_ERROR(kLogModule, "failed to load/compile OpenVINO model {} on {}: {}", model_path_,
			          device_, exception.what());
		}
		catch(...)
		{
			ready_ = false;
			LOG_ERROR(kLogModule, "failed to load/compile OpenVINO model {} on {}: unknown error",
			          model_path_, device_);
		}
	}

	bool OpenVINOInference::is_ready() const noexcept
	{
		return ready_;
	}

	std::vector<RawDetection> OpenVINOInference::infer(const cv::Mat& image)
	{
		if(image.empty())
		{
			LOG_WARN(kLogModule, "empty image passed to infer");
			return {};
		}

		if(!ready_)
		{
			throw std::runtime_error("OpenVINO inference is not ready");
		}

		// ---- 预处理：letterbox（左上角对齐、右侧/底部填充）----
		const double scale = detector_detail::letterbox_scale(
		    image.rows, image.cols, static_cast<int>(input_h_), static_cast<int>(input_w_));

		const cv::Size scaled_size = detector_detail::letterbox_size(image.rows, image.cols, scale);

		cv::Mat input(static_cast<int>(input_h_), static_cast<int>(input_w_), CV_8UC3,
		              cv::Scalar(0, 0, 0));

		const cv::Rect roi(0, 0, scaled_size.width, scaled_size.height);

		cv::resize(image, input(roi), scaled_size);

		ov::Tensor input_tensor(ov::element::u8, {1, input_h_, input_w_, 3}, input.data);

		// ---- 推理 ----
		auto infer_request = compiled_model_.create_infer_request();
		infer_request.set_input_tensor(input_tensor);
		infer_request.infer();

		const ov::Tensor output_tensor = infer_request.get_output_tensor();

		// ---- 输出协议校验 ----
		const ov::Shape output_shape = output_tensor.get_shape();

		if(output_shape.size() != 3)
		{
			throw std::runtime_error(
			    std::format("unexpected output rank {} (expected 3)", output_shape.size()));
		}

		if(output_shape[2] != detector_detail::kYoloV5RowWidth)
		{
			throw std::runtime_error(std::format(
			    "unexpected output row width {} (expected {})", output_shape[2],
			    detector_detail::kYoloV5RowWidth));
		}

		if(output_tensor.get_element_type() != ov::element::f32)
		{
			throw std::runtime_error("unexpected output element type (expected f32)");
		}

		const std::size_t row_count = output_shape[1];
		const float* data = output_tensor.data<float>();

		// ---- 解码 ----
		std::vector<RawDetection> detections;

		for(std::size_t row_index = 0; row_index < row_count; ++row_index)
		{
			const auto decoded = detector_detail::decode_row(
			    data + row_index * detector_detail::kYoloV5RowWidth, scale, confidence_threshold_);

			if(decoded.accepted)
			{
				detections.emplace_back(decoded.detection);
			}
		}

		return detections;
	}

} // namespace app::auto_aim