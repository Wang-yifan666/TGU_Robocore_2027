#include "app/auto_aim/detector/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "tools/logger.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "DETECTOR";

		using Clock = std::chrono::steady_clock;

		double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end)
		{
			return std::chrono::duration<double, std::milli>(end - begin).count();
		}

		bool is_finite_point(const cv::Point2f& point)
		{
			return std::isfinite(point.x) && std::isfinite(point.y);
		}

		bool is_valid_raw_detection(const RawDetection& detection)
		{
			if(!std::isfinite(detection.confidence))
			{
				return false;
			}

			for(const auto& point: detection.keypoints)
			{
				if(!is_finite_point(point))
				{
					return false;
				}
			}

			return true;
		}

		cv::Rect make_bounding_box(const std::array<cv::Point2f, 4>& keypoints)
		{
			const std::vector<cv::Point2f> points(keypoints.begin(), keypoints.end());

			return cv::boundingRect(points);
		}

		double calculate_iou(const cv::Rect& lhs, const cv::Rect& rhs)
		{
			const cv::Rect intersection = lhs & rhs;

			const double intersection_area = static_cast<double>(std::max(intersection.area(), 0));

			if(intersection_area <= 0.0)
			{
				return 0.0;
			}

			const double lhs_area = static_cast<double>(std::max(lhs.area(), 0));

			const double rhs_area = static_cast<double>(std::max(rhs.area(), 0));

			const double union_area = lhs_area + rhs_area - intersection_area;

			if(union_area <= 0.0)
			{
				return 0.0;
			}

			return intersection_area / union_area;
		}

		std::vector<Armor> apply_nms(std::vector<Armor> armors, float nms_threshold)
		{
			if(armors.size() <= 1)
			{
				return armors;
			}

			std::vector<std::size_t> indices(armors.size());

			std::iota(indices.begin(), indices.end(), std::size_t{0});

			std::stable_sort(indices.begin(), indices.end(),
			                 [&armors](std::size_t lhs, std::size_t rhs) {
				                 return armors[lhs].confidence > armors[rhs].confidence;
			                 });

			std::vector<bool> suppressed(armors.size(), false);

			std::vector<std::size_t> kept_indices;
			kept_indices.reserve(armors.size());

			for(std::size_t i = 0; i < indices.size(); ++i)
			{
				const std::size_t current_index = indices[i];

				if(suppressed[current_index])
				{
					continue;
				}

				kept_indices.push_back(current_index);

				for(std::size_t j = i + 1; j < indices.size(); ++j)
				{
					const std::size_t candidate_index = indices[j];

					if(suppressed[candidate_index])
					{
						continue;
					}

					const double iou =
					    calculate_iou(armors[current_index].box, armors[candidate_index].box);

					if(iou > static_cast<double>(nms_threshold))
					{
						suppressed[candidate_index] = true;
					}
				}
			}

			std::vector<Armor> result;
			result.reserve(kept_indices.size());

			for(const std::size_t index: kept_indices)
			{
				result.emplace_back(std::move(armors[index]));
			}

			return result;
		}

	} // namespace

	Detector::Detector(DetectorConfig config, std::unique_ptr<Inference> inference):
	config_(std::move(config)), inference_(std::move(inference))
	{
	}

	bool Detector::is_ready() const noexcept
	{
		return inference_ != nullptr && inference_->is_ready();
	}

	DetectionResult Detector::detect(const cv::Mat& image, std::uint64_t frame_id)
	{
		DetectionResult result;
		result.frame_id = frame_id;

		if(image.empty())
		{
			LOG_WARN(kLogModule, "frame {} is empty", frame_id);

			return result;
		}

		if(!is_ready())
		{
			LOG_WARN(kLogModule, "detector is not ready for frame {}", frame_id);

			return result;
		}

		std::vector<RawDetection> raw_detections;

		const auto inference_begin = Clock::now();

		try
		{
			raw_detections = inference_->infer(image);
		}
		catch(const std::exception& exception)
		{
			const auto inference_end = Clock::now();

			result.inference_time_ms = elapsed_ms(inference_begin, inference_end);

			LOG_ERROR(kLogModule, "inference failed on frame {}: {}", frame_id, exception.what());

			return result;
		}
		catch(...)
		{
			const auto inference_end = Clock::now();

			result.inference_time_ms = elapsed_ms(inference_begin, inference_end);

			LOG_ERROR(kLogModule,
			          "inference failed on frame {}: "
			          "unknown exception",
			          frame_id);

			return result;
		}

		const auto inference_end = Clock::now();

		result.inference_time_ms = elapsed_ms(inference_begin, inference_end);

		const auto postprocess_begin = Clock::now();

		std::vector<Armor> candidates;
		candidates.reserve(raw_detections.size());

		for(const auto& detection: raw_detections)
		{
			if(!is_valid_raw_detection(detection))
			{
				LOG_DEBUG(kLogModule,
				          "invalid raw detection skipped "
				          "on frame {}",
				          frame_id);

				continue;
			}

			if(detection.confidence < config_.confidence_threshold)
			{
				continue;
			}

			Armor armor = convert_detection(detection);

			if(!is_valid_armor(armor))
			{
				continue;
			}

			candidates.emplace_back(std::move(armor));
		}

		result.armors = apply_nms(std::move(candidates), config_.nms_threshold);

		const auto postprocess_end = Clock::now();

		result.postprocess_time_ms = elapsed_ms(postprocess_begin, postprocess_end);

		LOG_DEBUG(kLogModule,
		          "frame {}: raw={}, armors={}, "
		          "inference={:.3f} ms, postprocess={:.3f} ms",
		          frame_id, raw_detections.size(), result.armors.size(), result.inference_time_ms,
		          result.postprocess_time_ms);

		return result;
	}

	bool Detector::is_valid_armor(const Armor& armor) const
	{
		if(armor.points.size() != 4)
		{
			return false;
		}

		for(const auto& point: armor.points)
		{
			if(!is_finite_point(point))
			{
				return false;
			}
		}

		if(!std::isfinite(armor.confidence))
		{
			return false;
		}

		if(armor.confidence < config_.confidence_threshold)
		{
			return false;
		}

		if(armor.color != config_.enemy_color)
		{
			return false;
		}

		if(armor.name == ArmorName::NotArmor)
		{
			return false;
		}

		if(armor.type == ArmorType::Unknown)
		{
			return false;
		}

		if(!std::isfinite(armor.ratio))
		{
			return false;
		}

		if(armor.ratio < config_.min_armor_ratio)
		{
			return false;
		}

		if(armor.ratio > config_.max_armor_ratio)
		{
			return false;
		}

		if(!std::isfinite(armor.rectangular_error))
		{
			return false;
		}

		if(armor.rectangular_error > config_.max_rectangular_error)
		{
			return false;
		}

		if(armor.box.width <= 0 || armor.box.height <= 0)
		{
			return false;
		}

		return true;
	}

	Armor Detector::convert_detection(const RawDetection& detection) const
	{
		std::vector<cv::Point2f> keypoints(detection.keypoints.begin(), detection.keypoints.end());

		const cv::Rect box = make_bounding_box(detection.keypoints);

		return Armor(detection.color_id, detection.number_id, detection.confidence, box,
		             std::move(keypoints));
	}

} // namespace app::auto_aim
