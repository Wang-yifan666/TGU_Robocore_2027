/**
 * @file test_auto_aim.cpp
 * @brief AutoAim facade 单元测试（FakeInference + 真实 Detector + synthetic Solver）。
 *
 * 覆盖：
 * - empty frame -> NoFrame
 * - dependencies not ready -> Error
 * - zero detections -> NoTarget
 * - one valid detection -> Detecting
 * - multiple detections 目标排序（pre-tracker 确定性选择）
 * - PnP failure -> NoTarget（单块 PnP 失败不视为 fatal Error）
 * - successful solved target -> xyz_in_gimbal finite, distance > 0
 * - reset / frame counter 行为
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector.hpp"
#include "app/auto_aim/solver.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

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
		// 合成输入尺寸：宽 1280，高 720；图像中心 (640, 360)。
		return cv::Mat(720, 1280, CV_8UC3, cv::Scalar::all(0));
	}

	cv::Mat make_normal_camera_matrix()
	{
		return (cv::Mat_<double>(3, 3) << 800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0);
	}

	auto_aim::SolverConfig make_valid_solver_config()
	{
		auto_aim::SolverConfig config;

		config.camera_matrix = make_normal_camera_matrix();
		config.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);

		config.r_gimbal_to_imu_body = Eigen::Matrix3d::Identity();
		config.r_camera_to_gimbal = Eigen::Matrix3d::Identity();
		config.t_camera_to_gimbal = Eigen::Vector3d::Zero();

		config.lightbar_length_m = 56e-3;
		config.small_armor_width_m = 135e-3;
		config.big_armor_width_m = 230e-3;

		return config;
	}

	// 退化相机矩阵（3x3 全零）能通过 is_valid()，但 solvePnP 会产生非有限结果，
	// 被 Solver 防御性校验拒绝 -> 确定性 PnP failure。
	auto_aim::SolverConfig make_degenerate_solver_config()
	{
		auto_aim::SolverConfig config;

		config.camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
		config.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);

		config.r_gimbal_to_imu_body = Eigen::Matrix3d::Identity();
		config.r_camera_to_gimbal = Eigen::Matrix3d::Identity();
		config.t_camera_to_gimbal = Eigen::Vector3d::Zero();

		config.lightbar_length_m = 56e-3;
		config.small_armor_width_m = 135e-3;
		config.big_armor_width_m = 230e-3;

		return config;
	}

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

	auto_aim::RawDetection make_valid_blue_three(const std::array<cv::Point2f, 4>& keypoints)
	{
		return make_detection(keypoints, 0, 3, 0.90F);
	}

	auto_aim::AutoAim build_auto_aim(std::vector<auto_aim::RawDetection> detections,
	                                 const auto_aim::SolverConfig& solver_config,
	                                 bool detector_ready = true)
	{
		auto fake = std::make_unique<FakeInference>(std::move(detections), detector_ready);

		auto_aim::Detector detector(make_default_config(), std::move(fake));

		auto_aim::Solver solver(solver_config);

		return auto_aim::AutoAim(std::move(detector), std::move(solver));
	}

	bool is_finite_point(const cv::Point2f& point)
	{
		return std::isfinite(point.x) && std::isfinite(point.y);
	}

	// 靠近图像中心 (640, 360) 的直立小装甲板。
	const std::array<cv::Point2f, 4> kNearKeypoints = {{
	    cv::Point2f{600.0F, 340.0F}, cv::Point2f{680.0F, 340.0F}, cv::Point2f{680.0F, 400.0F},
	    cv::Point2f{600.0F, 400.0F}}};

	// 远离图像中心 (640, 360) 的直立小装甲板。
	const std::array<cv::Point2f, 4> kFarKeypoints = {{
	    cv::Point2f{100.0F, 100.0F}, cv::Point2f{180.0F, 100.0F}, cv::Point2f{180.0F, 160.0F},
	    cv::Point2f{100.0F, 160.0F}}};

	// ============================================================
	// 测试：依赖未就绪 -> Error
	// ============================================================

	void test_dependencies_not_ready(TestRunner& runner)
	{
		runner.begin("Dependencies not ready");

		// 1) Detector 未就绪（FakeInference ready=false）。
		{
			auto auto_aim = build_auto_aim({}, make_valid_solver_config(), /*detector_ready=*/false);

			runner.expect(!auto_aim.is_ready(), "AutoAim should report not ready");

			auto_aim::FrameContext frame;
			frame.image = make_test_image();

			const auto result = auto_aim.process(frame);

			runner.expect(result.state == auto_aim::AimState::Error,
			              "Unready dependencies should produce Error state");
			runner.expect(!result.has_target, "Error result should have no target");
		}

		// 2) Solver 非法（空配置 -> is_valid() == false）。
		{
			auto_aim::SolverConfig invalid_config;

			auto auto_aim = build_auto_aim({}, invalid_config);

			runner.expect(!auto_aim.is_ready(), "AutoAim with invalid solver should be not ready");

			auto_aim::FrameContext frame;
			frame.image = make_test_image();

			const auto result = auto_aim.process(frame);

			runner.expect(result.state == auto_aim::AimState::Error,
			              "Invalid solver should produce Error state");
			runner.expect(!result.has_target, "Error result should have no target");
		}

		runner.end();
	}

	// ============================================================
	// 测试：空帧 -> NoFrame
	// ============================================================

	void test_empty_frame(TestRunner& runner)
	{
		runner.begin("Empty frame");

		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_valid_solver_config());

		runner.expect(auto_aim.is_ready(), "AutoAim should be ready");

		auto_aim::FrameContext frame;

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::NoFrame,
		              "Empty frame should produce NoFrame state");
		runner.expect(!result.has_target, "NoFrame result should have no target");

		runner.end();
	}

	// ============================================================
	// 测试：零检测 -> NoTarget
	// ============================================================

	void test_zero_detections(TestRunner& runner)
	{
		runner.begin("Zero detections");

		auto auto_aim = build_auto_aim({}, make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::NoTarget,
		              "Zero detections should produce NoTarget state");
		runner.expect(!result.has_target, "NoTarget result should have no target");

		runner.end();
	}

	// ============================================================
	// 测试：单块有效检测 -> Detecting
	// ============================================================

	void test_one_valid_detection(TestRunner& runner)
	{
		runner.begin("One valid detection");

		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::Detecting,
		              "One valid detection should produce Detecting state");
		runner.expect(result.has_target, "Detecting result should have a target");

		if(result.has_target)
		{
			runner.expect(result.target.name == auto_aim::ArmorName::Three,
			              "Target name should be three");
			runner.expect(result.target.type == auto_aim::ArmorType::Small,
			              "Target type should be small");
			runner.expect(result.target.color == auto_aim::ArmorColor::Blue,
			              "Target color should be blue");

			runner.expect(result.target.points.size() == 4,
			              "Target should keep four keypoints");

			for(const auto& point: result.target.points)
			{
				runner.expect(is_finite_point(point), "Target 2D points should be finite");
			}

			const bool finite_xyz =
			    std::isfinite(result.target.xyz_in_gimbal.x())
			    && std::isfinite(result.target.xyz_in_gimbal.y())
			    && std::isfinite(result.target.xyz_in_gimbal.z());

			runner.expect(finite_xyz, "Target xyz_in_gimbal should be finite");

			runner.expect(result.distance > 0.0,
			              "Solved target distance should be positive");
		}

		runner.end();
	}

	// ============================================================
	// 测试：多目标排序（pre-tracker 确定性选择）
	// ============================================================

	void test_multiple_detections_ordering(TestRunner& runner)
	{
		runner.begin("Multiple detections ordering");

		// 当前 Detector 管线不会填充 ArmorPriority（YOLO-id 构造不设置 priority），
		// 因此所有候选优先级相等时，按距离图像中心越近越优先。
		auto auto_aim = build_auto_aim(
		    {make_valid_blue_three(kFarKeypoints), make_valid_blue_three(kNearKeypoints)},
		    make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::Detecting,
		              "Multiple detections should produce Detecting state");
		runner.expect(result.has_target, "Multiple detections should yield a target");

		if(result.has_target)
		{
			// 靠近图像中心的目标中心 = (640, 370)。
			const cv::Point2f expected_near_center(640.0F, 370.0F);

			const double error = cv::norm(result.target.center - expected_near_center);

			runner.expect(error < 1e-3,
			              "Closer-to-image-center armor should be selected preferentially");
		}

		runner.end();
	}

	// ============================================================
	// 测试：PnP 失败 -> NoTarget
	// ============================================================

	void test_pnp_failure(TestRunner& runner)
	{
		runner.begin("PnP failure");

		// 检测本身几何有效，但退化相机矩阵导致 solvePnP 输出非有限结果，
		// 被 Solver 防御性校验拒绝 -> AutoAim 返回 NoTarget，而非 fatal Error。
		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_degenerate_solver_config());

		runner.expect(auto_aim.is_ready(),
		              "AutoAim should be considered ready (solver is_valid on 3x3 non-empty)");

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::NoTarget,
		              "PnP failure should produce NoTarget (not fatal Error)");
		runner.expect(!result.has_target, "PnP failure result should have no target");

		runner.end();
	}

	// ============================================================
	// 测试：reset / frame counter 行为
	// ============================================================

	void test_reset_frame_counter(TestRunner& runner)
	{
		runner.begin("Reset / frame counter");

		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		// 连续处理若干帧（含无目标帧），不应崩溃。
		const auto result1 = auto_aim.process(frame);
		const auto result2 = auto_aim.process(frame);

		runner.expect(result1.state == auto_aim::AimState::Detecting,
		              "First processed frame should detect");
		runner.expect(result2.state == auto_aim::AimState::Detecting,
		              "Second processed frame should detect");

		// reset 后重新处理，仍能正常检测（内部帧计数已归零）。
		auto_aim.reset();

		const auto result_after_reset = auto_aim.process(frame);

		runner.expect(result_after_reset.state == auto_aim::AimState::Detecting,
		              "Process after reset should still detect");

		runner.end();
	}

} // namespace

int main()
{
	std::printf("=== AutoAim Module Test Suite ===\n\n");

	TestRunner runner;

	test_dependencies_not_ready(runner);
	test_empty_frame(runner);
	test_zero_detections(runner);
	test_one_valid_detection(runner);
	test_multiple_detections_ordering(runner);
	test_pnp_failure(runner);
	test_reset_frame_counter(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== AutoAim tests failed ===\n");
		return 1;
	}

	std::printf("=== All auto_aim tests passed ===\n");
	return 0;
}