#include "app/auto_aim/detector/detector.hpp"
#include "app/auto_aim/detector/inference.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

	// ============================================================
	// 简单测试运行器
	// ============================================================

	class TestRunner
	{
	public:
		void begin(std::string_view name)
		{
			current_test_ = name;
			current_test_failed_ = false;

			std::printf("===== %.*s =====\n", static_cast<int>(name.size()), name.data());
		}

		void expect(bool condition, std::string_view message)
		{
			++check_count_;

			if(condition)
			{
				std::printf("[PASS] %.*s\n", static_cast<int>(message.size()), message.data());

				return;
			}

			++failure_count_;
			current_test_failed_ = true;

			std::printf("[FAIL] %.*s\n", static_cast<int>(message.size()), message.data());
		}

		void end()
		{
			std::printf("[%s] %.*s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
			            static_cast<int>(current_test_.size()), current_test_.data());
		}

		[[nodiscard]] int failure_count() const noexcept
		{
			return failure_count_;
		}

		void print_summary() const
		{
			std::printf("========================================\n");
			std::printf("Checks:   %d\n", check_count_);
			std::printf("Failures: %d\n", failure_count_);
			std::printf("========================================\n");
		}

	private:
		std::string_view current_test_;

		int check_count_ = 0;
		int failure_count_ = 0;

		bool current_test_failed_ = false;
	};

	// ============================================================
	// FakeInference
	// ============================================================

	class FakeInference final: public auto_aim::Inference
	{
	public:
		explicit FakeInference(std::vector<auto_aim::RawDetection> detections = {},
		                       bool ready = true): detections_(std::move(detections)), ready_(ready)
		{
		}

		[[nodiscard]] bool is_ready() const noexcept override
		{
			return ready_;
		}

		std::vector<auto_aim::RawDetection> infer(const cv::Mat& image) override
		{
			++infer_call_count_;
			last_image_size_ = image.size();

			return detections_;
		}

		void set_detections(std::vector<auto_aim::RawDetection> detections)
		{
			detections_ = std::move(detections);
		}

		[[nodiscard]] int infer_call_count() const noexcept
		{
			return infer_call_count_;
		}

		[[nodiscard]] cv::Size last_image_size() const noexcept
		{
			return last_image_size_;
		}

	private:
		std::vector<auto_aim::RawDetection> detections_;

		bool ready_ = true;

		int infer_call_count_ = 0;
		cv::Size last_image_size_;
	};

	// ============================================================
	// 测试数据构造
	// ============================================================

	auto_aim::DetectorConfig make_default_config()
	{
		auto_aim::DetectorConfig config;

		config.enemy_color = auto_aim::ArmorColor::Blue;

		config.inference_score_threshold = 0.70F;
		config.min_confidence = 0.80F;
		config.nms_threshold = 0.30F;

		config.min_armor_ratio = 1.0;
		config.max_armor_ratio = 6.0;
		config.max_rectangular_error = 0.50;

		config.enable_debug = false;

		return config;
	}

	cv::Mat make_test_image()
	{
		return cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0));
	}


	// 更改测试时的敌人
	auto_aim::RawDetection make_detection(const std::array<cv::Point2f, 4>& keypoints,
	                                      int color_id = 0, int number_id = 3,
	                                      float confidence = 0.90F)
	{
		auto_aim::RawDetection detection;

		detection.keypoints = keypoints;
		detection.color_id = color_id;
		detection.number_id = number_id;
		detection.confidence = confidence;

		return detection;
	}

	auto_aim::RawDetection make_valid_blue_three()
	{
		return make_detection(
		    {
		        cv::Point2f(100.0F, 100.0F),
		        cv::Point2f(160.0F, 100.0F),
		        cv::Point2f(160.0F, 140.0F),
		        cv::Point2f(100.0F, 140.0F),
		    },
		    0, 3, 0.90F);
	}

	bool near(double lhs, double rhs, double tolerance = 1e-6)
	{
		return std::abs(lhs - rhs) <= tolerance;
	}

	// ============================================================
	// 测试：正常检测
	// ============================================================

	void test_valid_detection(TestRunner& runner)
	{
		runner.begin("Valid detection");

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    make_valid_blue_three(),
		});

		auto* fake_ptr = fake.get();

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		runner.expect(detector.is_ready(), "Detector should be ready");

		const auto image = make_test_image();
		const auto result = detector.detect(image, 42);

		runner.expect(fake_ptr->infer_call_count() == 1, "Inference should be called once");

		runner.expect(fake_ptr->last_image_size() == image.size(),
		              "Inference should receive the original image");

		runner.expect(result.frame_id == 42, "Frame id should be preserved");

		runner.expect(result.armors.size() == 1, "One valid armor should be returned");

		runner.expect(result.inference_time_ms >= 0.0, "Inference time should be non-negative");

		runner.expect(result.postprocess_time_ms >= 0.0, "Postprocess time should be non-negative");

		if(result.armors.size() == 1)
		{
			const auto& armor = result.armors.front();

			runner.expect(armor.color == auto_aim::ArmorColor::Blue, "Armor color should be blue");

			runner.expect(armor.name == auto_aim::ArmorName::Three, "Armor name should be three");

			runner.expect(armor.type == auto_aim::ArmorType::Small, "Armor type should be small");

			runner.expect(near(armor.confidence, 0.90), "Armor confidence should be preserved");

			runner.expect(armor.points.size() == 4, "Armor should contain four keypoints");

			runner.expect(cv::norm(armor.center - cv::Point2f(130.0F, 120.0F)) < 1e-3,
			              "Armor center should be calculated correctly");

			runner.expect(near(armor.ratio, 1.5, 1e-3),
			              "Armor ratio should be calculated correctly");

			runner.expect(armor.rectangular_error < 1e-3,
			              "Rectangular armor should have small geometry error");
		}

		runner.end();
	}

	// ============================================================
	// 测试：敌方颜色过滤
	// ============================================================

	void test_enemy_color_filter(TestRunner& runner)
	{
		runner.begin("Enemy color filter");

		auto config = make_default_config();
		config.enemy_color = auto_aim::ArmorColor::Red;

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    make_valid_blue_three(),
		});

		auto_aim::Detector detector(config, std::move(fake));

		const auto result = detector.detect(make_test_image(), 1);

		runner.expect(result.armors.empty(),
		              "Blue armor should be rejected when enemy color is red");

		runner.end();
	}

	// ============================================================
	// 测试：低置信度过滤
	// ============================================================

	void test_confidence_filter(TestRunner& runner)
	{
		runner.begin("Confidence filter");

		auto detection = make_valid_blue_three();
		detection.confidence = 0.20F;

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    detection,
		});

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const auto result = detector.detect(make_test_image(), 2);

		runner.expect(result.armors.empty(), "Low-confidence detection should be rejected");

		runner.end();
	}

	// ============================================================
	// 测试：比例异常
	// ============================================================

	void test_ratio_filter(TestRunner& runner)
	{
		runner.begin("Armor ratio filter");

		const auto detection = make_detection({
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(400.0F, 100.0F),
		    cv::Point2f(400.0F, 120.0F),
		    cv::Point2f(100.0F, 120.0F),
		});

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    detection,
		});

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const auto result = detector.detect(make_test_image(), 3);

		runner.expect(result.armors.empty(),
		              "Armor with excessive width-height ratio should be rejected");

		runner.end();
	}

	// ============================================================
	// 测试：矩形误差异常
	// ============================================================

	void test_rectangular_error_filter(TestRunner& runner)
	{
		runner.begin("Rectangular error filter");

		auto config = make_default_config();

		// 放宽比例限制，使本测试只关注 rectangular_error。
		config.min_armor_ratio = 0.1;
		config.max_armor_ratio = 20.0;
		config.max_rectangular_error = 0.20;

		const auto detection = make_detection({
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(220.0F, 100.0F),
		    cv::Point2f(120.0F, 180.0F),
		    cv::Point2f(100.0F, 180.0F),
		});

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    detection,
		});

		auto_aim::Detector detector(config, std::move(fake));

		const auto result = detector.detect(make_test_image(), 4);

		runner.expect(result.armors.empty(),
		              "Armor with excessive rectangular error should be rejected");

		runner.end();
	}

	// ============================================================
	// 测试：非法坐标
	// ============================================================

	void test_invalid_keypoint_coordinates(TestRunner& runner)
	{
		runner.begin("Invalid keypoint coordinates");

		const float nan = std::numeric_limits<float>::quiet_NaN();

		const auto detection = make_detection({
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(nan, 100.0F),
		    cv::Point2f(160.0F, 140.0F),
		    cv::Point2f(100.0F, 140.0F),
		});

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    detection,
		});

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const auto result = detector.detect(make_test_image(), 5);

		runner.expect(result.armors.empty(),
		              "Detection containing NaN coordinates should be rejected");

		runner.end();
	}

	// ============================================================
	// 测试：退化四边形
	// ============================================================

	void test_degenerate_keypoints(TestRunner& runner)
	{
		runner.begin("Degenerate keypoints");

		const auto detection = make_detection({
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(100.0F, 100.0F),
		    cv::Point2f(100.0F, 100.0F),
		});

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    detection,
		});

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const auto result = detector.detect(make_test_image(), 6);

		runner.expect(result.armors.empty(), "Degenerate keypoints should be rejected");

		runner.end();
	}

	// ============================================================
	// 测试：空图像
	// ============================================================

	void test_empty_image(TestRunner& runner)
	{
		runner.begin("Empty image");

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    make_valid_blue_three(),
		});

		auto* fake_ptr = fake.get();

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const cv::Mat empty_image;

		const auto result = detector.detect(empty_image, 7);

		runner.expect(result.armors.empty(), "Empty image should produce no armors");

		runner.expect(fake_ptr->infer_call_count() == 0,
		              "Inference should not run for an empty image");

		runner.expect(result.frame_id == 7, "Frame id should still be preserved");

		runner.end();
	}

	// ============================================================
	// 测试：推理器未就绪
	// ============================================================

	void test_inference_not_ready(TestRunner& runner)
	{
		runner.begin("Inference not ready");

		auto fake = std::make_unique<FakeInference>(
		    std::vector<auto_aim::RawDetection>{
		        make_valid_blue_three(),
		    },
		    false);

		auto* fake_ptr = fake.get();

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		runner.expect(!detector.is_ready(), "Detector should report not ready");

		const auto result = detector.detect(make_test_image(), 8);

		runner.expect(result.armors.empty(), "Unready detector should produce no armors");

		runner.expect(fake_ptr->infer_call_count() == 0, "Unready inference should not be called");

		runner.end();
	}

	// ============================================================
	// 测试：空推理器指针
	// ============================================================

	void test_null_inference(TestRunner& runner)
	{
		runner.begin("Null inference");

		auto_aim::Detector detector(make_default_config(), nullptr);

		runner.expect(!detector.is_ready(), "Detector with null inference should not be ready");

		const auto result = detector.detect(make_test_image(), 9);

		runner.expect(result.armors.empty(), "Detector with null inference should return safely");

		runner.end();
	}

	// ============================================================
	// 测试：混合检测结果
	// ============================================================

	void test_mixed_detections(TestRunner& runner)
	{
		runner.begin("Mixed detections");

		auto valid = make_valid_blue_three();

		auto low_confidence = make_valid_blue_three();
		low_confidence.confidence = 0.10F;

		auto red_armor = make_valid_blue_three();
		red_armor.color_id = 1;

		auto invalid_ratio = make_detection({
		    cv::Point2f(50.0F, 50.0F),
		    cv::Point2f(550.0F, 50.0F),
		    cv::Point2f(550.0F, 60.0F),
		    cv::Point2f(50.0F, 60.0F),
		});

		auto fake = std::make_unique<FakeInference>(std::vector<auto_aim::RawDetection>{
		    valid,
		    low_confidence,
		    red_armor,
		    invalid_ratio,
		});

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		const auto result = detector.detect(make_test_image(), 10);

		runner.expect(result.armors.size() == 1, "Only one detection should survive all filters");

		if(result.armors.size() == 1)
		{
			runner.expect(result.armors.front().name == auto_aim::ArmorName::Three,
			              "Remaining armor should be the valid blue three");
		}

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_detector");
	std::printf("=== Detector Module Test Suite ===\n\n");

	TestRunner runner;

	test_valid_detection(runner);
	test_enemy_color_filter(runner);
	test_confidence_filter(runner);
	test_ratio_filter(runner);
	test_rectangular_error_filter(runner);
	test_invalid_keypoint_coordinates(runner);
	test_degenerate_keypoints(runner);
	test_empty_image(runner);
	test_inference_not_ready(runner);
	test_null_inference(runner);
	test_mixed_detections(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Detector tests failed ===\n");
		return 1;
	}

	std::printf("=== All detector tests passed ===\n");
	return 0;
}
