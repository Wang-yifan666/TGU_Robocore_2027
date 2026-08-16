/**
 * @file test_auto_aim_pipeline.cpp
 * @brief demo 视频离线集成测试：真实 OpenVINO + Detector + Solver + AutoAim。
 *
 * 数据流：
 *   data/demo/demo.avi
 *     -> OpenVINOInference
 *     -> Detector
 *     -> pre-tracker 目标选择
 *     -> Solver (demo-only 标定)
 *     -> AutoAim::process(FrameContext)
 *     -> AimResult
 *
 * 姿态数据：
 * - 若存在与 demo.avi 严格同步的 data/demo/demo.txt（格式：timestamp w x y z），
 *   则按"一帧图像对应一行 quaternion"读取，并传入 FrameContext。
 * - 否则 fallback 到 Eigen::Quaterniond::Identity()，并输出警告：
 *     "No synchronized demo quaternion data found. World-frame results are not validated."
 *
 * 注意：本测试不验证真实 world 坐标，也不做"每帧必须检测到目标"的断言。
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace
{

	constexpr const char* MODULE = "AUTO_AIM_PIPELINE";

	struct PipelineStats
	{
		std::size_t processed_frames = 0;
		std::size_t detected_frames = 0;   // has_target == true
		std::size_t no_target_frames = 0;  // NoTarget
		std::size_t error_frames = 0;      // Error / NoFrame（异常路径）
	};

	bool is_finite_point(const cv::Point2f& point)
	{
		return std::isfinite(point.x) && std::isfinite(point.y);
	}

	bool is_valid_state(app::auto_aim::AimState state)
	{
		switch(state)
		{
		case app::auto_aim::AimState::Idle:
		case app::auto_aim::AimState::NoFrame:
		case app::auto_aim::AimState::NoTarget:
		case app::auto_aim::AimState::Detecting:
		case app::auto_aim::AimState::Tracking:
		case app::auto_aim::AimState::TargetLocked:
		case app::auto_aim::AimState::Error:
			return true;
		default:
			return false;
		}
	}

	/**
	 * @brief 尝试加载与视频同步的 quaternion 文件（timestamp w x y z）。
	 * @return true 成功加载；false 文件不存在/格式非法。
	 */
	bool try_load_quaternions(const std::string& path,
	                          std::vector<std::tuple<double, Eigen::Quaterniond>>& out)
	{
		std::ifstream file(path);
		if(!file.is_open())
		{
			return false;
		}

		double timestamp = 0.0;
		double w = 0.0, x = 0.0, y = 0.0, z = 0.0;
		std::size_t count = 0;

		while(file >> timestamp >> w >> x >> y >> z)
		{
			Eigen::Quaterniond q(w, x, y, z);
			q.normalize();
			out.emplace_back(timestamp, q);
			++count;
		}

		if(count == 0)
		{
			return false;
		}

		std::printf("[%s] loaded %zu synchronized quaternions from %s\n", MODULE, count,
		            path.c_str());
		return true;
	}

} // namespace

int main()
{
	// ============================================================
	// 1. 配置路径
	// ============================================================
	const std::string project_root = std::string(PROJECT_SOURCE_DIR) + "/";

	const std::string video_path = project_root + "data/demo/demo.avi";
	const std::string quaternion_path = project_root + "data/demo/demo.txt";
	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";

	// ============================================================
	// 2. demo 视频与 quaternion 数据检查
	// ============================================================
	cv::VideoCapture video(video_path);
	if(!video.isOpened())
	{
		std::printf("[%s] [ERROR] failed to open video: %s\n", MODULE, video_path.c_str());
		return 1;
	}

	double fps = video.get(cv::CAP_PROP_FPS);
	if(!std::isfinite(fps) || fps <= 0.0)
	{
		fps = 30.0;
	}

	std::vector<std::tuple<double, Eigen::Quaterniond>> quaternions;
	const bool has_sync_quaternions = try_load_quaternions(quaternion_path, quaternions);

	if(!has_sync_quaternions)
	{
		std::printf("[%s] [WARN] No synchronized demo quaternion data found.\n"
		            "[%s] [WARN] World-frame results are not validated.\n"
		            "[%s] [WARN] Falling back to Eigen::Quaterniond::Identity().\n",
		            MODULE, MODULE, MODULE);
	}

	// ============================================================
	// 3. 配置加载（demo 使用 demo-only solver 标定）
	// ============================================================
	app::auto_aim::DetectorConfig detector_config;
	if(!app::auto_aim::load_detector_config(detector_config_path, detector_config))
	{
		std::printf("[%s] [ERROR] failed to load detector config: %s\n", MODULE,
		            detector_config_path.c_str());
		return 2;
	}

	app::auto_aim::SolverConfig solver_config;
	if(!app::auto_aim::load_solver_config(solver_config_path, solver_config))
	{
		std::printf("[%s] [ERROR] failed to load solver config: %s\n", MODULE,
		            solver_config_path.c_str());
		return 3;
	}

	// ============================================================
	// 4. 构建真实 OpenVINO 推理后端 + Detector + Solver + AutoAim
	// ============================================================
	auto inference = std::make_unique<app::auto_aim::OpenVINOInference>(
	    detector_config.model_path, detector_config.device,
	    detector_config.inference_score_threshold);

	if(!inference->is_ready())
	{
		std::printf("[%s] [ERROR] OpenVINO inference is not ready\n", MODULE);
		return 4;
	}

	app::auto_aim::Detector detector(detector_config, std::move(inference));
	app::auto_aim::Solver solver(solver_config);
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver));

	if(!auto_aim.is_ready())
	{
		std::printf("[%s] [ERROR] AutoAim dependencies are not ready\n", MODULE);
		return 5;
	}

	// ============================================================
	// 5. 逐帧处理直到 EOF
	// ============================================================
	PipelineStats stats;

	cv::Mat frame;
	std::size_t frame_index = 0;

	while(true)
	{
		if(!video.read(frame))
		{
			break;
		}

		if(frame.empty())
		{
			++stats.error_frames;
			break;
		}

		app::auto_aim::FrameContext context;
		context.image = frame;
		context.timestamp_s = static_cast<double>(frame_index) / fps;

		if(has_sync_quaternions)
		{
			if(frame_index >= quaternions.size())
			{
				std::printf("[%s] [WARN] video frame %zu exceeds quaternion data size %zu; "
				            "synchronization mismatch detected\n",
				            MODULE, frame_index, quaternions.size());
				break;
			}

			context.q_imu_body_to_world = std::get<1>(quaternions[frame_index]);
		}
		// 无同步数据时保持 Identity（上面已输出警告）。

		const auto result = auto_aim.process(context);

		if(!is_valid_state(result.state))
		{
			std::printf("[%s] [ERROR] illegal AimState %d at frame %zu\n", MODULE,
			            static_cast<int>(result.state), frame_index);
			return 6;
		}

		++stats.processed_frames;

		if(result.state == app::auto_aim::AimState::NoTarget)
		{
			++stats.no_target_frames;
		}
		else if(result.state == app::auto_aim::AimState::Detecting && result.has_target)
		{
			++stats.detected_frames;

			// 弱断言：成功结果必须是有限且有正距离。
			if(!std::isfinite(result.distance) || result.distance <= 0.0)
			{
				std::printf("[%s] [ERROR] non-positive/non-finite distance %.6f at frame %zu\n",
				            MODULE, result.distance, frame_index);
				return 7;
			}

			if(!std::isfinite(result.target.xyz_in_gimbal.x())
			   || !std::isfinite(result.target.xyz_in_gimbal.y())
			   || !std::isfinite(result.target.xyz_in_gimbal.z()))
			{
				std::printf("[%s] [ERROR] non-finite xyz_in_gimbal at frame %zu\n", MODULE,
				            frame_index);
				return 8;
			}

			for(const auto& point: result.target.points)
			{
				if(!is_finite_point(point))
				{
					std::printf("[%s] [ERROR] non-finite 2D point at frame %zu\n", MODULE,
					            frame_index);
					return 9;
				}
			}

			// 统计输出（每帧记录，见下方汇总；此处保留最近一帧详情供调试）。
			if(frame_index % 60 == 0 || frame_index == 0)
			{
				std::printf(
				    "[%s] frame %zu: state=%d has_target=%d name=%s type=%s color=%s "
				    "conf=%.3f xyz_g=({%.3f},{%.3f},{%.3f}) xyz_w=({%.3f},{%.3f},{%.3f}) "
				    "dist=%.3f\n",
				    MODULE, frame_index, static_cast<int>(result.state), (int)result.has_target,
				    std::string(app::auto_aim::to_string(result.target.name)).c_str(),
				    std::string(app::auto_aim::to_string(result.target.type)).c_str(),
				    std::string(app::auto_aim::to_string(result.target.color)).c_str(),
				    result.target.confidence, result.target.xyz_in_gimbal.x(),
				    result.target.xyz_in_gimbal.y(), result.target.xyz_in_gimbal.z(),
				    result.target.xyz_in_world.x(), result.target.xyz_in_world.y(),
				    result.target.xyz_in_world.z(), result.distance);
			}
		}
		else if(result.state == app::auto_aim::AimState::Error
		        || result.state == app::auto_aim::AimState::NoFrame)
		{
			++stats.error_frames;
		}

		++frame_index;
	}

	video.release();

	// 同步数据一方提前结束：视频帧数少于 quaternion 行数时，也报告不匹配。
	if(has_sync_quaternions && frame_index < quaternions.size())
	{
		std::printf("[%s] [WARN] quaternion data has %zu entries but video ended at frame %zu; "
		            "synchronization mismatch detected\n",
		            MODULE, quaternions.size(), frame_index);
	}

	// ============================================================
	// 6. 汇总与弱断言
	// ============================================================
	std::printf("\n================= Pipeline Summary =================\n");
	std::printf("video:           %s\n", video_path.c_str());
	std::printf("processed_frames:%zu\n", stats.processed_frames);
	std::printf("detected_frames: %zu\n", stats.detected_frames);
	std::printf("no_target_frames:%zu\n", stats.no_target_frames);
	std::printf("error_frames:    %zu\n", stats.error_frames);
	std::printf("sync_quaternion: %s\n", has_sync_quaternions ? "yes" : "no (identity fallback)");
	std::printf("===================================================\n");

	if(stats.processed_frames == 0)
	{
		std::printf("[%s] [ERROR] no frames processed\n", MODULE);
		return 10;
	}

	std::printf("[%s] demo pipeline finished successfully\n", MODULE);
	return 0;
}