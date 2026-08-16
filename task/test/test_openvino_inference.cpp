/**
 * @file test_openvino_inference.cpp
 * @brief OpenVINOInference 纯数学单测 + 可选 OpenVINO smoke test。
 */
#include "app/auto_aim/detector/openvino_inference.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

namespace
{
	namespace auto_aim = app::auto_aim;
	namespace detail = app::auto_aim::detector_detail;

	// ============================================================
	// 简单测试运行器（支持 PASS / FAIL / SKIP）
	// ============================================================

	class TestRunner
	{
	public:
		void begin(std::string_view name)
		{
			current_test_ = name;
			current_test_failed_ = false;
			current_test_skipped_ = false;

			std::printf("===== %.*s =====\n", static_cast<int>(name.size()), name.data());
		}

		void expect(bool condition, std::string_view message)
		{
			if(condition)
			{
				++pass_count_;
				std::printf("[PASS] %.*s\n", static_cast<int>(message.size()), message.data());
				return;
			}

			++failure_count_;
			current_test_failed_ = true;

			std::printf("[FAIL] %.*s\n", static_cast<int>(message.size()), message.data());
		}

		void skip(std::string_view reason)
		{
			++skip_count_;
			current_test_skipped_ = true;

			std::printf("[SKIP] %s\n", std::string(reason).c_str());
		}

		void end()
		{
			if(current_test_skipped_)
			{
				std::printf("[SKIPPED] %.*s\n\n", static_cast<int>(current_test_.size()),
				            current_test_.data());
			}
			else
			{
				std::printf("[%s] %.*s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
				            static_cast<int>(current_test_.size()), current_test_.data());
			}
		}

		[[nodiscard]] int failure_count() const noexcept
		{
			return failure_count_;
		}

		[[nodiscard]] int skip_count() const noexcept
		{
			return skip_count_;
		}

		void print_summary() const
		{
			std::printf("========================================\n");
			std::printf("Passed:    %d\n", pass_count_);
			std::printf("Failed:    %d\n", failure_count_);
			std::printf("Skipped:   %d\n", skip_count_);
			std::printf("========================================\n");
		}

	private:
		std::string_view current_test_;

		int pass_count_ = 0;
		int failure_count_ = 0;
		int skip_count_ = 0;

		bool current_test_failed_ = false;
		bool current_test_skipped_ = false;
	};

	bool near(double lhs, double rhs, double tolerance = 1e-6)
	{
		return std::abs(lhs - rhs) <= tolerance;
	}

	// ============================================================
	// letterbox 参数计算
	// ============================================================

	void test_letterbox_scale(TestRunner& runner)
	{
		runner.begin("letterbox scale");

		// 1280x720 -> 640x640
		runner.expect(near(detail::letterbox_scale(720, 1280, 640, 640), 0.5),
		              "1280x720 scale should be 0.5");

		// 1280x1024 -> 640x640
		runner.expect(near(detail::letterbox_scale(1024, 1280, 640, 640), 0.5),
		              "1280x1024 scale should be 0.5");

		// 640x480 -> 640x640
		runner.expect(near(detail::letterbox_scale(480, 640, 640, 640), 1.0),
		              "640x480 scale should be 1.0");

		// 竖图 480x640（宽<高，宽度受限）
		runner.expect(near(detail::letterbox_scale(640, 480, 640, 640), 1.0),
		              "480x640 (portrait) scale should be 1.0 (width-limited)");

		// 正方形 640x640
		runner.expect(near(detail::letterbox_scale(640, 640, 640, 640), 1.0),
		              "640x640 scale should be 1.0");

		runner.end();
	}

	void test_letterbox_size(TestRunner& runner)
	{
		runner.begin("letterbox size");

		auto size = detail::letterbox_size(720, 1280, 0.5);
		runner.expect(size.width == 640 && size.height == 360, "1280x720 -> 640x360");

		size = detail::letterbox_size(1024, 1280, 0.5);
		runner.expect(size.width == 640 && size.height == 512, "1280x1024 -> 640x512");

		size = detail::letterbox_size(480, 640, 1.0);
		runner.expect(size.width == 640 && size.height == 480, "640x480 -> 640x480");

		size = detail::letterbox_size(640, 640, 1.0);
		runner.expect(size.width == 640 && size.height == 640, "640x640 -> 640x640");

		runner.end();
	}

	void test_map_point(TestRunner& runner)
	{
		runner.begin("map point to original");

		const auto mapped = detail::map_point_to_original(cv::Point2f(320.0F, 180.0F), 0.5);

		runner.expect(near(mapped.x, 640.0) && near(mapped.y, 360.0), "(320,180)/0.5 -> (640,360)");

		const auto origin = detail::map_point_to_original(cv::Point2f(0.0F, 0.0F), 0.5);
		runner.expect(near(origin.x, 0.0) && near(origin.y, 0.0), "(0,0)/0.5 -> (0,0)");

		runner.end();
	}

	void test_decode_yolov5_row(TestRunner& runner)
	{
		runner.begin("decode yolov5 row");

		float row[detail::kYoloV5RowWidth] = {};

		// 关键点：kp0(10,20), kp1(30,40), kp2(50,60), kp3(70,80)
		row[0] = 10.0F;
		row[1] = 20.0F;
		row[2] = 70.0F;
		row[3] = 80.0F;
		row[4] = 50.0F;
		row[5] = 60.0F;
		row[6] = 30.0F;
		row[7] = 40.0F;

		// objectness logit = 0 -> sigmoid = 0.5，恰好通过阈值 0.5
		row[8] = 0.0F;

		// 颜色 one-hot：颜色索引 1 最大
		row[9] = 0.1F;
		row[10] = 0.9F;
		row[11] = 0.2F;
		row[12] = 0.3F;

		// 编号 one-hot：编号索引 3 最大
		row[13] = 0.1F;
		row[14] = 0.2F;
		row[15] = 0.3F;
		row[16] = 0.9F;
		row[17] = 0.1F;
		row[18] = 0.1F;
		row[19] = 0.1F;
		row[20] = 0.1F;
		row[21] = 0.1F;

		const auto decoded = detail::decode_yolov5_row(row, 0.5, 0.5F);

		runner.expect(decoded.accepted, "row should be accepted");

		runner.expect(near(decoded.detection.confidence, 0.5),
		              "confidence should be sigmoid(0)=0.5");

		runner.expect(decoded.detection.color_id == 1, "color_id should be 1");
		runner.expect(decoded.detection.number_id == 3, "number_id should be 3");

		// 关键点顺序固定为 {col0/1, col6/7, col4/5, col2/3}，每个 / scale=0.5
		const auto& kp = decoded.detection.keypoints;

		runner.expect(near(kp[0].x, 20.0) && near(kp[0].y, 40.0),
		              "kp[0] should be {col0,col1}/scale = (20,40)");
		runner.expect(near(kp[1].x, 60.0) && near(kp[1].y, 80.0),
		              "kp[1] should be {col6,col7}/scale = (60,80)");
		runner.expect(near(kp[2].x, 100.0) && near(kp[2].y, 120.0),
		              "kp[2] should be {col4,col5}/scale = (100,120)");
		runner.expect(near(kp[3].x, 140.0) && near(kp[3].y, 160.0),
		              "kp[3] should be {col2,col3}/scale = (140,160)");

		runner.end();
	}

	void test_decode_yolov5_row_low_confidence(TestRunner& runner)
	{
		runner.begin("decode yolov5 row low confidence");

		float row[detail::kYoloV5RowWidth] = {};
		row[8] = -10.0F; // sigmoid -> 接近 0

		const auto decoded = detail::decode_yolov5_row(row, 0.5, 0.5F);

		runner.expect(!decoded.accepted, "low objectness row should be rejected");

		runner.end();
	}

	void test_decode_yolov5_row_nan(TestRunner& runner)
	{
		runner.begin("decode yolov5 row NaN");

		float row[detail::kYoloV5RowWidth] = {};
		row[0] = std::numeric_limits<float>::quiet_NaN();

		const auto decoded = detail::decode_yolov5_row(row, 0.5, 0.5F);

		runner.expect(!decoded.accepted, "row containing NaN should be rejected");

		runner.end();
	}

	// ============================================================
	// OpenVINO smoke test（有模型时执行）
	// ============================================================

	void test_openvino_smoke(TestRunner& runner)
	{
		runner.begin("OpenVINO smoke");

#ifdef PROJECT_SOURCE_DIR
		const std::filesystem::path model_path =
		    std::filesystem::path(PROJECT_SOURCE_DIR) / "data/models/armor.xml";
#else
		const std::filesystem::path model_path = "data/models/armor.xml";
#endif

		if(!std::filesystem::exists(model_path))
		{
			runner.skip("model not found");
			runner.end();
			return;
		}

		auto_aim::OpenVINOInference inference(model_path.string(), "CPU", 0.70F);

		runner.expect(inference.is_ready(), "OpenVINOInference should be ready after load");

		if(!inference.is_ready())
		{
			runner.end();
			return;
		}

		cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));

		try
		{
			const auto detections = inference.infer(image);

			for(const auto& detection: detections)
			{
				runner.expect(std::isfinite(detection.confidence), "confidence should be finite");

				for(const auto& point: detection.keypoints)
				{
					runner.expect(std::isfinite(point.x) && std::isfinite(point.y),
					              "keypoint should be finite");
				}

				runner.expect(detection.color_id >= 0 && detection.color_id <= 3,
				              "color_id should be in [0,3]");

				runner.expect(detection.number_id >= 0 && detection.number_id <= 8,
				              "number_id should be in [0,8]");
			}

			std::printf("[INFO] smoke produced %zu detections\n", detections.size());
		}
		catch(const std::exception& exception)
		{
			runner.expect(false, "infer should not throw");
			std::printf("[ERROR] %s\n", exception.what());
		}

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== OpenVINO Inference Test Suite ===\n\n");

	TestRunner runner;

	test_letterbox_scale(runner);
	test_letterbox_size(runner);
	test_map_point(runner);
	test_decode_yolov5_row(runner);
	test_decode_yolov5_row_low_confidence(runner);
	test_decode_yolov5_row_nan(runner);
	test_openvino_smoke(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== OpenVINO inference tests failed ===\n");
		return 1;
	}

	std::printf("=== OpenVINO inference tests passed (skipped=%d) ===\n", runner.skip_count());
	return 0;
}