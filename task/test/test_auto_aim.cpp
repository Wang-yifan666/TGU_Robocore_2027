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
 * - 退化相机矩阵（fx/fy=0）-> Solver 非法，AutoAim 不应 ready
 * - successful solved target -> xyz_in_gimbal finite, distance > 0
 * - reset / frame counter 行为
 * - debug 模式与非 debug 模式算法行为等价（双实例对比）
 * - debug 生命周期：early return / 跨帧残留均必须清零
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector.hpp"
#include "app/auto_aim/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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

	// 逐帧返回不同检测结果的 FakeInference（测试 debug 生命周期用）。
	class QueueFakeInference final: public auto_aim::Inference
	{
	public:
		explicit QueueFakeInference(std::vector<std::vector<auto_aim::RawDetection>> frames,
		                            bool ready = true): frames_(std::move(frames)), ready_(ready)
		{
		}

		[[nodiscard]] bool is_ready() const noexcept override
		{
			return ready_;
		}

		std::vector<auto_aim::RawDetection> infer(const cv::Mat& image) override
		{
			(void)image;

			if(index_ < frames_.size())
			{
				return frames_[index_++];
			}

			return {};
		}

	private:
		std::vector<std::vector<auto_aim::RawDetection>> frames_;

		bool ready_ = true;

		std::size_t index_ = 0;
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

	// 退化相机矩阵（3x3 全零）：fx / fy 均为 0，
	// Solver 构造时 valid_ 应判定为 false，AutoAim 不应视为 ready。
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

		auto_aim::Tracker tracker(auto_aim::make_default_tracker_config());

		return auto_aim::AutoAim(std::move(detector), std::move(solver), std::move(tracker));
	}

	// 用假数据填满 debug，验证 empty-frame / error 路径会将其清空。
	void fill_debug_with_fake_data(auto_aim::AutoAimDebugData& debug)
	{
		auto_aim::Armor fake_armor;
		fake_armor.name = auto_aim::ArmorName::Three;
		fake_armor.type = auto_aim::ArmorType::Small;
		fake_armor.color = auto_aim::ArmorColor::Blue;
		fake_armor.confidence = 0.99F;
		fake_armor.points = {cv::Point2f(0.0F, 0.0F), cv::Point2f(10.0F, 0.0F),
		                     cv::Point2f(10.0F, 20.0F), cv::Point2f(0.0F, 20.0F)};

		debug.detected_armors = {fake_armor};
		debug.selected_armor_index = 0;
		debug.solved_armor_indices = {0};
		debug.inference_time_ms = 12.34;
		debug.postprocess_time_ms = 5.67;
	}

	bool is_finite_point(const cv::Point2f& point)
	{
		return std::isfinite(point.x) && std::isfinite(point.y);
	}

	bool is_finite_vector3(const Eigen::Vector3d& vector)
	{
		return std::isfinite(vector.x()) && std::isfinite(vector.y()) && std::isfinite(vector.z());
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
		// P1-1：Detecting 阶段 has_target == false，但 tracked_target 可用。
		runner.expect(!result.has_target, "Detecting result should have has_target == false");
		runner.expect(result.tracked_target.has_value(),
		              "Detecting result should still expose tracked_target");
		runner.expect(result.tracked_target
		                  && result.tracked_target->state == auto_aim::TrackerState::Detecting,
		              "tracked_target state should be Detecting");

		// legacy visible-armor 兼容输出仍在。
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
		runner.expect(!result.has_target,
		              "Multiple detections should yield has_target == false (Detecting)");

		// legacy 可见装甲板选择仍应先靠近图像中心者。
		const cv::Point2f expected_near_center(640.0F, 370.0F);
		const double error = cv::norm(result.target.center - expected_near_center);
		runner.expect(error < 1e-3,
		              "Closer-to-image-center armor should be selected preferentially");

		runner.end();
	}

	// ============================================================
	// 测试：退化相机矩阵 -> Solver 非法，AutoAim 不应 ready
	// ============================================================

	void test_degenerate_solver_not_ready(TestRunner& runner)
	{
		runner.begin("Degenerate solver not ready");

		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_degenerate_solver_config());

		runner.expect(!auto_aim.is_ready(),
		              "Zero-fx/fy camera matrix should make Solver invalid, AutoAim not ready");

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		const auto result = auto_aim.process(frame);

		runner.expect(result.state == auto_aim::AimState::Error,
		              "Unready AutoAim should produce Error state");
		runner.expect(!result.has_target, "Error result should have no target");

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

	// ============================================================
	// 测试：debug 模式与非 debug 模式算法行为等价
	// ============================================================
	//
	// 必须使用两个初始状态完全相同的独立 AutoAim 实例，
	// 分别调用 process(frame) 和 process(frame, &debug)，
	// 禁止用同一个实例连续调用两次（避免 frame counter 差异污染）。

	void test_debug_behavior_equivalence(TestRunner& runner)
	{
		runner.begin("Debug behavior equivalence");

		auto_aim::SolverConfig solver_config = make_valid_solver_config();

		// 两个检测：远 + 近，触发 pre-tracker ordering 分支。
		const std::vector<auto_aim::RawDetection> detections = {
		    make_valid_blue_three(kFarKeypoints), make_valid_blue_three(kNearKeypoints)};

		auto lhs = build_auto_aim(detections, solver_config);
		auto rhs = build_auto_aim(detections, solver_config);

		auto_aim::FrameContext frame;
		frame.image = make_test_image();
		frame.timestamp_s = 12.5;

		const auto result_plain = lhs.process(frame);

		auto_aim::AutoAimDebugData debug;
		const auto result_debug = rhs.process(frame, &debug);

		// 核心结果必须完全一致。
		runner.expect(result_plain.state == result_debug.state,
		              "AimState should be identical with/without debug");
		runner.expect(result_plain.has_target == result_debug.has_target,
		              "has_target should be identical with/without debug");
		runner.expect(std::abs(result_plain.distance - result_debug.distance) < 1e-9,
		              "distance should be identical with/without debug");
		runner.expect(std::abs(result_plain.timestamp_s - result_debug.timestamp_s) < 1e-12,
		              "timestamp should be identical with/without debug");

		if(result_plain.has_target)
		{
			runner.expect(result_plain.target.name == result_debug.target.name,
			              "Target name should be identical");
			runner.expect(result_plain.target.type == result_debug.target.type,
			              "Target type should be identical");
			runner.expect(result_plain.target.color == result_debug.target.color,
			              "Target color should be identical");
			runner.expect(result_plain.target.center == result_debug.target.center,
			              "Target center should be identical");
			runner.expect(result_plain.target.points.size() == result_debug.target.points.size(),
			              "Target points count should be identical");

			if(result_plain.target.points.size() == result_debug.target.points.size())
			{
				for(std::size_t i = 0; i < result_plain.target.points.size(); ++i)
				{
					runner.expect(result_plain.target.points[i] == result_debug.target.points[i],
					              "Target 2D points should be identical");
				}
			}

			const double xyz_error =
			    (result_plain.target.xyz_in_gimbal - result_debug.target.xyz_in_gimbal).norm();
			runner.expect(xyz_error < 1e-9,
			              "Target xyz_in_gimbal should be identical with/without debug");
		}

		// observations 数量与内容必须一致（debug 仅旁路观察，不应改变观测）。
		runner.expect(result_plain.observations.size() == result_debug.observations.size(),
		              "observations count should be identical with/without debug");

		const std::size_t observation_count = std::min(result_plain.observations.size(),
		                                               result_debug.observations.size());

		for(std::size_t i = 0; i < observation_count; ++i)
		{
			const auto& lhs_observation = result_plain.observations[i];
			const auto& rhs_observation = result_debug.observations[i];

			runner.expect(lhs_observation.source_detection_index
			                  == rhs_observation.source_detection_index,
			              "observation source_detection_index should be identical");
			runner.expect(std::abs(lhs_observation.timestamp_s - rhs_observation.timestamp_s)
			                  < 1e-12,
			              "observation timestamp should be identical");

			const double gimbal_error =
			    (lhs_observation.position_in_gimbal - rhs_observation.position_in_gimbal).norm();
			runner.expect(gimbal_error < 1e-9,
			              "observation position_in_gimbal should be identical");

			const double world_error =
			    (lhs_observation.position_in_world - rhs_observation.position_in_world).norm();
			runner.expect(world_error < 1e-9,
			              "observation position_in_world should be identical");

			runner.expect(std::abs(lhs_observation.armor_yaw_in_world
			                       - rhs_observation.armor_yaw_in_world)
			                  < 1e-9,
			              "observation armor_yaw_in_world should be identical");
		}

		// 近目标（原始顺序 index 1）应被选中，且 debug 下标指向同一块装甲板。
		if(debug.selected_armor_index.has_value())
		{
			runner.expect(*debug.selected_armor_index == 1,
			              "Near-center armor (original index 1) should be selected");
			runner.expect(debug.detected_armors.size() == 2,
			              "Debug should expose both detected armors");
			runner.expect(cv::norm(debug.detected_armors[*debug.selected_armor_index].center
			                           - result_plain.target.center)
			                  < 1e-3,
			              "Debug selected index should point to the same armor as AimResult");

			// debug 中的检测结果不应被 Solver 污染：空间坐标仍为 0。
			const double zero_norm =
			    debug.detected_armors[*debug.selected_armor_index].xyz_in_gimbal.norm();
			runner.expect(zero_norm < 1e-12,
			              "Debug detected armors should be pre-Solver (xyz untouched)");
		}
		else
		{
			runner.expect(false, "Debug should record a selected armor index");
		}

		runner.end();
	}

	// ============================================================
	// 测试：debug 生命周期 —— early return 必须清空 debug
	// ============================================================

	void test_debug_lifecycle_early_return(TestRunner& runner)
	{
		runner.begin("Debug lifecycle on early return");

		// 1) Error 路径：依赖未就绪。
		{
			auto auto_aim = build_auto_aim({}, make_valid_solver_config(), /*detector_ready=*/false);

			auto_aim::FrameContext frame;
			frame.image = make_test_image();

			auto_aim::AutoAimDebugData debug;
			fill_debug_with_fake_data(debug);

			const auto result = auto_aim.process(frame, &debug);

			runner.expect(result.state == auto_aim::AimState::Error,
			              "Unready dependencies should produce Error state");
			runner.expect(debug.detected_armors.empty(), "Error path must clear detected_armors");
			runner.expect(!debug.selected_armor_index.has_value(),
			              "Error path must clear selected_armor_index");
			runner.expect(debug.solved_armor_indices.empty(),
			              "Error path must clear solved_armor_indices");
			runner.expect(debug.inference_time_ms == 0.0 && debug.postprocess_time_ms == 0.0,
			              "Error path must clear timing");
		}

		// 2) NoFrame 路径：空帧。
		{
			auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
			                               make_valid_solver_config());

			auto_aim::FrameContext frame; // image 保持 empty

			auto_aim::AutoAimDebugData debug;
			fill_debug_with_fake_data(debug);

			const auto result = auto_aim.process(frame, &debug);

			runner.expect(result.state == auto_aim::AimState::NoFrame,
			              "Empty frame should produce NoFrame state");
			runner.expect(debug.detected_armors.empty(), "NoFrame path must clear detected_armors");
			runner.expect(!debug.selected_armor_index.has_value(),
			              "NoFrame path must clear selected_armor_index");
			runner.expect(debug.solved_armor_indices.empty(),
			              "NoFrame path must clear solved_armor_indices");
			runner.expect(debug.inference_time_ms == 0.0 && debug.postprocess_time_ms == 0.0,
			              "NoFrame path must clear timing");
		}

		runner.end();
	}

	// ============================================================
	// 测试：单 observation（1 个 detection，PnP 成功）
	// ============================================================

	void test_single_observation(TestRunner& runner)
	{
		runner.begin("Single observation");

		auto auto_aim = build_auto_aim({make_valid_blue_three(kNearKeypoints)},
		                               make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();
		frame.timestamp_s = 12.345;

		auto_aim::AutoAimDebugData debug;
		const auto result = auto_aim.process(frame, &debug);

		runner.expect(result.state == auto_aim::AimState::Detecting,
		              "Single detection should produce Detecting state");
		runner.expect(result.observations.size() == 1,
		              "Single successful PnP should produce exactly one observation");

		if(!result.observations.empty())
		{
			const auto& observation = result.observations[0];

			runner.expect(observation.source_detection_index == 0,
			              "Single observation should trace to detection index 0");
			runner.expect(observation.timestamp_s == 12.345,
			              "Observation timestamp should come from FrameContext (not default 0)");
			runner.expect(observation.name == auto_aim::ArmorName::Three,
			              "Observation name should be three");
			runner.expect(observation.type == auto_aim::ArmorType::Small,
			              "Observation type should be small");
			runner.expect(observation.color == auto_aim::ArmorColor::Blue,
			              "Observation color should be blue");

			runner.expect(std::abs(observation.confidence - 0.90F) < 1e-6,
			              "Observation confidence should match detection confidence");

			runner.expect(is_finite_vector3(observation.position_in_gimbal),
			              "Observation position_in_gimbal should be finite");
			runner.expect(is_finite_vector3(observation.position_in_world),
			              "Observation position_in_world should be finite");
			runner.expect(is_finite_vector3(observation.ypd_in_world),
			              "Observation ypd_in_world should be finite");
			runner.expect(std::isfinite(observation.armor_yaw_in_world),
			              "Observation armor_yaw_in_world should be finite");
		}

		runner.end();
	}

	// ============================================================
	// 测试：多 observation（2 个 detection，全部 PnP 成功）
	// ============================================================

	void test_multiple_observations(TestRunner& runner)
	{
		runner.begin("Multiple observations");

		// 输入顺序：far(0), near(1)。
		auto auto_aim = build_auto_aim(
		    {make_valid_blue_three(kFarKeypoints), make_valid_blue_three(kNearKeypoints)},
		    make_valid_solver_config());

		auto_aim::FrameContext frame;
		frame.image = make_test_image();
		frame.timestamp_s = 7.25;

		auto_aim::AutoAimDebugData debug;
		const auto result = auto_aim.process(frame, &debug);

		runner.expect(result.state == auto_aim::AimState::Detecting,
		              "Multiple detections should produce Detecting state");
		runner.expect(result.observations.size() == 2,
		              "Two successful PnP detections should produce two observations");

		if(result.observations.size() == 2)
		{
			// observations 保持 Detector 原始顺序（远在前、近在后）。
			runner.expect(result.observations[0].source_detection_index == 0,
			              "First observation should trace to detection index 0 (far)");
			runner.expect(result.observations[1].source_detection_index == 1,
			              "Second observation should trace to detection index 1 (near)");

			for(const auto& observation: result.observations)
			{
				runner.expect(is_finite_vector3(observation.position_in_gimbal),
				              "Each observation position_in_gimbal should be finite");
				runner.expect(std::isfinite(observation.armor_yaw_in_world),
				              "Each observation armor_yaw_in_world should be finite");
			}
		}

		// observation 顺序 != pre-tracker target 选择顺序：
		// observations[0] 是远目标(index 0)，而 target 选择近目标(index 1)。
		runner.expect(debug.selected_armor_index.has_value()
		                  && *debug.selected_armor_index == 1,
		              "Pre-tracker target selection should still pick near armor (index 1)");

		{
			const cv::Point2f expected_near_center(640.0F, 370.0F);
			runner.expect(cv::norm(result.target.center - expected_near_center) < 1e-3,
			              "Selected target should still be closer-to-center armor");
		}

		// ---- debug invariant：solved_armor_indices 与 observations 一致 ----
		runner.expect(debug.solved_armor_indices.size() == result.observations.size(),
		              "solved_armor_indices size should equal observations size");

		const std::size_t count = std::min(debug.solved_armor_indices.size(),
		                                   result.observations.size());

		bool indices_match = true;

		for(std::size_t i = 0; i < count; ++i)
		{
			if(debug.solved_armor_indices[i] != result.observations[i].source_detection_index)
			{
				indices_match = false;
				break;
			}
		}

		runner.expect(indices_match,
		              "solved_armor_indices should match observations source_detection_index");

		runner.end();
	}

	// ============================================================
	// 测试：debug 生命周期 —— 上一帧有检测、下一帧无检测不得残留
	// ============================================================

	void test_debug_lifecycle_no_residue(TestRunner& runner)
	{
		runner.begin("Debug lifecycle no residue across frames");

		std::vector<std::vector<auto_aim::RawDetection>> frame_detections;
		frame_detections.push_back({make_valid_blue_three(kNearKeypoints)}); // 有检测
		frame_detections.push_back({});                                      // 无检测

		auto fake = std::make_unique<QueueFakeInference>(std::move(frame_detections));
		auto_aim::Detector detector(make_default_config(), std::move(fake));
		auto_aim::Solver solver(make_valid_solver_config());
		auto_aim::Tracker tracker(auto_aim::make_default_tracker_config());
		auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));

		auto_aim::FrameContext frame;
		frame.image = make_test_image();

		auto_aim::AutoAimDebugData debug;

		const auto result_first = auto_aim.process(frame, &debug);

		runner.expect(result_first.state == auto_aim::AimState::Detecting,
		              "First frame with detection should detect");
		runner.expect(debug.detected_armors.size() == 1,
		              "First frame debug should expose the detected armor");
		runner.expect(debug.selected_armor_index.has_value(),
		              "First frame debug should record selected index");
		runner.expect(result_first.observations.size() == 1,
		              "First frame should produce one observation");
		runner.expect(debug.solved_armor_indices.size() == 1,
		              "First frame debug should record one solved index");

		const auto result_second = auto_aim.process(frame, &debug);

		// Commit 6 起：无检测帧会驱动 Tracker miss，但目标仍处于 Detecting
		// （detecting_max_misses 未超），因此不是 NoTarget。
		runner.expect(result_second.state == auto_aim::AimState::Detecting,
		              "Second frame without detection drives Tracker miss, stays Detecting");
		runner.expect(result_second.observations.empty(),
		              "Second frame result must not retain previous frame observations");
		runner.expect(debug.detected_armors.empty(),
		              "Second frame must not retain previous frame armors");
		runner.expect(!debug.selected_armor_index.has_value(),
		              "Second frame must not retain previous frame selected index");
		runner.expect(debug.solved_armor_indices.empty(),
		              "Second frame must not retain previous frame solved indices");

		// NoTarget 帧仍然执行了 detect()，因此 timing 是当前帧的全新数据
		// （允许非零），而不是上一帧的残留；残留只针对 armors / index。
		// Error / NoFrame 路径的 timing 必须严格为 0 已由
		// test_debug_lifecycle_early_return 覆盖。
		runner.expect(std::isfinite(debug.inference_time_ms)
		                  && debug.inference_time_ms >= 0.0
		                  && std::isfinite(debug.postprocess_time_ms)
		                  && debug.postprocess_time_ms >= 0.0,
		              "Second frame timing should be fresh current-frame data");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_auto_aim");
	std::printf("=== AutoAim Module Test Suite ===\n\n");

	TestRunner runner;

	test_dependencies_not_ready(runner);
	test_empty_frame(runner);
	test_zero_detections(runner);
	test_one_valid_detection(runner);
	test_multiple_detections_ordering(runner);
	test_degenerate_solver_not_ready(runner);
	test_reset_frame_counter(runner);
	test_debug_behavior_equivalence(runner);
	test_single_observation(runner);
	test_multiple_observations(runner);
	test_debug_lifecycle_early_return(runner);
	test_debug_lifecycle_no_residue(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== AutoAim tests failed ===\n");
		return 1;
	}

	std::printf("=== All auto_aim tests passed ===\n");
	return 0;
}