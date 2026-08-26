/**
 * @file test_aimer_visual.cpp
 * @brief 在 demo.avi 上可视化 Aimer 瞄准解（人工回放，不注册 CTest）。
 *
 * 数据流：
 *   data/demo/demo.avi (+ demo.txt 同步 quaternion)
 *     -> OpenVINOInference -> Detector -> Solver(全部) -> AutoAim
 *     -> AimResult::tracked_target
 *     -> Aimer::aim(tracked_target, t_now, bullet_speed, &debug)
 *     -> AimingSolution + AimerDebugData
 *     -> 世界坐标反投影到图像 -> overlay
 *
 * 绘制内容：
 * - Detector 输出的装甲板四点轮廓（青色）
 * - Tracker 车辆中心（绿色实心点 + 十字）
 * - 车辆速度向量（洋红箭头）
 * - 各预测 armor hypothesis（品红圆点 + armor_id；选中的装甲板高亮为红色粗圈）
 * - Aimer 瞄准点（黄色十字 + 圆圈，来自 debug.aim_point_in_world）
 * - Aimer 状态文本：status / yaw / pitch / selected_armor_id / fire_allowed /
 *   flight_time / converged / refinement_iterations
 *
 * 无 ground truth：只做可视化 + 数值健康诊断，不声称 absolute accuracy。
 */

#include "app/auto_aim/aimer.hpp"
#include "app/auto_aim/aimer_config.hpp"
#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"
#include "app/auto_aim/tracker_config.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "test_logging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

	constexpr const char* kWindowName = "aimer_visual";

	constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

	// BGR 颜色常量。
	const cv::Scalar kDetectedArmorColor{255, 200, 0}; // 青
	const cv::Scalar kCenterColor{0, 255, 0};          // 绿
	const cv::Scalar kHypothesisColor{255, 0, 255};    // 品红
	const cv::Scalar kSelectedColor{0, 0, 255};        // 红
	const cv::Scalar kVelocityColor{255, 0, 255};      // 洋红
	const cv::Scalar kAimPointColor{0, 255, 255};      // 黄
	const cv::Scalar kTextColor{255, 255, 255};

	struct SyncQuaternion
	{
		double timestamp_s = 0.0;
		Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
	};

	bool try_load_quaternions(const std::string& path, std::vector<SyncQuaternion>& out)
	{
		std::ifstream file(path);

		if(!file.is_open())
		{
			return false;
		}

		double timestamp = 0.0;
		double w = 0.0, x = 0.0, y = 0.0, z = 0.0;

		while(file >> timestamp >> w >> x >> y >> z)
		{
			if(!std::isfinite(timestamp) || !std::isfinite(w) || !std::isfinite(x)
			   || !std::isfinite(y) || !std::isfinite(z))
			{
				return false;
			}

			const double norm2 = w * w + x * x + y * y + z * z;

			if(!std::isfinite(norm2) || norm2 <= 1e-12)
			{
				return false;
			}

			SyncQuaternion sync;
			sync.timestamp_s = timestamp;
			sync.q = Eigen::Quaterniond(w, x, y, z);
			sync.q.normalize();
			out.emplace_back(sync);
		}

		return !out.empty();
	}

	const char* tracker_state_name(app::auto_aim::TrackerState state)
	{
		switch(state)
		{
		case app::auto_aim::TrackerState::Lost:
			return "Lost";
		case app::auto_aim::TrackerState::Detecting:
			return "Detecting";
		case app::auto_aim::TrackerState::Tracking:
			return "Tracking";
		case app::auto_aim::TrackerState::TempLost:
			return "TempLost";
		default:
			return "Unknown";
		}
	}

	const char* aim_status_name(app::auto_aim::AimStatus status)
	{
		switch(status)
		{
		case app::auto_aim::AimStatus::Success:
			return "Success";
		case app::auto_aim::AimStatus::InvalidTarget:
			return "InvalidTarget";
		case app::auto_aim::AimStatus::InvalidBulletSpeed:
			return "InvalidBulletSpeed";
		case app::auto_aim::AimStatus::BallisticUnsolvable:
			return "BallisticUnsolvable";
		case app::auto_aim::AimStatus::NoValidArmor:
			return "NoValidArmor";
		case app::auto_aim::AimStatus::PredictionFailed:
			return "PredictionFailed";
		default:
			return "Unknown";
		}
	}

	/**
	 * @brief world 坐标（无全局平移的姿态稳定坐标系）反投影到图像。
	 *
	 * 与 Solver::solve 的坐标链一致：
	 *   gimbal = R_gimbal_to_world^T * world
	 *   camera = R_camera_to_gimbal^T * (gimbal - t_camera_to_gimbal)
	 * camera.z() <= 0（在相机后方）返回 nullopt。
	 */
	std::optional<cv::Point2f> project_world_to_image(const Eigen::Vector3d& world,
	                                                  const app::auto_aim::SolverConfig& solver,
	                                                  const Eigen::Quaterniond& q_imu_body_to_world)
	{
		const Eigen::Matrix3d r_imu_body_to_world = q_imu_body_to_world.normalized().toRotationMatrix();
		const Eigen::Matrix3d r_gimbal_to_world =
		    solver.r_gimbal_to_imu_body.transpose() * r_imu_body_to_world
		    * solver.r_gimbal_to_imu_body;

		const Eigen::Vector3d gimbal = r_gimbal_to_world.transpose() * world;
		const Eigen::Vector3d camera =
		    solver.r_camera_to_gimbal.transpose() * (gimbal - solver.t_camera_to_gimbal);

		if(!camera.allFinite() || camera.z() <= 0.0)
		{
			return std::nullopt;
		}

		cv::Mat k = solver.camera_matrix;
		k.convertTo(k, CV_64F);

		cv::Mat d = solver.distort_coeffs;

		if(!d.empty())
		{
			d.convertTo(d, CV_64F);
		}

		const std::vector<cv::Point3d> camera_point{cv::Point3d(camera.x(), camera.y(), camera.z())};
		std::vector<cv::Point2d> image_point;

		cv::projectPoints(camera_point, cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0), k, d,
		                  image_point);

		if(image_point.empty() || !std::isfinite(image_point[0].x)
		   || !std::isfinite(image_point[0].y))
		{
			return std::nullopt;
		}

		return cv::Point2f(static_cast<float>(image_point[0].x),
		                   static_cast<float>(image_point[0].y));
	}

	void draw_cross(cv::Mat& image, const cv::Point& center, const cv::Scalar& color, int radius,
	                int thickness = 2)
	{
		cv::line(image, cv::Point(center.x - radius, center.y),
		         cv::Point(center.x + radius, center.y), color, thickness);
		cv::line(image, cv::Point(center.x, center.y - radius),
		         cv::Point(center.x, center.y + radius), color, thickness);
	}

	std::vector<std::string> normalize_args(int argc, char** argv)
	{
		const std::vector<std::string> value_options = {"--video", "--quaternion", "--bullet-speed"};

		std::vector<std::string> normalized;

		for(int index = 0; index < argc; ++index)
		{
			const std::string current = argv[index];

			const bool is_value_option =
			    std::find(value_options.begin(), value_options.end(), current) != value_options.end();

			normalized.push_back(current);

			if(is_value_option && index + 1 < argc)
			{
				const std::string next = argv[index + 1];

				if(!next.empty() && next.rfind("--", 0) != 0)
				{
					normalized.back() = current + "=" + next;
					++index;
				}
			}
		}

		return normalized;
	}

} // namespace

int main(int argc, char** argv)
{
	if(!test_logging::init("test_aimer_visual"))
	{
		std::printf("[AIMER_VISUAL] [ERROR] logger init failed, file logging disabled\n");
	}

	const std::vector<std::string> normalized_args = normalize_args(argc, argv);

	std::vector<char*> arg_ptrs;

	for(const auto& arg: normalized_args)
	{
		arg_ptrs.push_back(const_cast<char*>(arg.c_str()));
	}

	const cv::String keys =
	    "{help h usage ? |      | print this help}"
	    "{video          |      | input video path (default data/demo/demo.avi)}"
	    "{quaternion     |      | synchronized quaternion file timestamp w x y z}"
	    "{bullet-speed   | 23.0 | bullet muzzle speed m/s}";

	cv::CommandLineParser parser(static_cast<int>(arg_ptrs.size()), arg_ptrs.data(), keys);

	if(parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const std::string project_root = std::string(PROJECT_SOURCE_DIR) + "/";

	std::string video_path = parser.get<std::string>("video");

	if(video_path.empty())
	{
		video_path = project_root + "data/demo/demo.avi";
	}

	std::string quaternion_path = parser.get<std::string>("quaternion");

	if(quaternion_path.empty())
	{
		quaternion_path = project_root + "data/demo/demo.txt";
	}

	const double bullet_speed_mps = parser.get<double>("bullet-speed");

	if(!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	LOG_INFO("AIMER_VISUAL", "starting visual replay: video={}, bullet_speed={:.2f} m/s",
	         video_path, bullet_speed_mps);

	cv::VideoCapture video(video_path);

	if(!video.isOpened())
	{
		std::printf("[AIMER_VISUAL] [ERROR] failed to open video: %s\n", video_path.c_str());
		return 2;
	}

	double fps = video.get(cv::CAP_PROP_FPS);

	if(!std::isfinite(fps) || fps <= 0.0)
	{
		fps = 30.0;
	}

	std::vector<SyncQuaternion> quaternions;
	const bool has_sync_quaternions = try_load_quaternions(quaternion_path, quaternions);

	if(!has_sync_quaternions)
	{
		std::printf("[AIMER_VISUAL] [WARN] no synchronized quaternion; world projection uses identity\n");
	}

	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";
	const std::string tracker_config_path = project_root + "config/app/auto_aim/tracker.toml";
	const std::string aimer_config_path = project_root + "config/app/auto_aim/aimer.toml";

	app::auto_aim::DetectorConfig detector_config;

	if(!app::auto_aim::load_detector_config(detector_config_path, detector_config))
	{
		std::printf("[AIMER_VISUAL] [ERROR] failed to load detector config\n");
		return 3;
	}

	app::auto_aim::SolverConfig solver_config;

	if(!app::auto_aim::load_solver_config(solver_config_path, solver_config))
	{
		std::printf("[AIMER_VISUAL] [ERROR] failed to load solver config\n");
		return 4;
	}

	app::auto_aim::TrackerConfig tracker_config;

	if(!app::auto_aim::load_tracker_config(tracker_config_path, tracker_config))
	{
		std::printf("[AIMER_VISUAL] [ERROR] failed to load tracker config\n");
		return 5;
	}

	app::auto_aim::AimerConfig aimer_config;

	if(!app::auto_aim::load_aimer_config(aimer_config_path, aimer_config))
	{
		std::printf("[AIMER_VISUAL] [ERROR] failed to load aimer config\n");
		return 6;
	}

	auto inference = std::make_unique<app::auto_aim::OpenVINOInference>(
	    detector_config.model_path, detector_config.device,
	    detector_config.inference_score_threshold);

	if(!inference->is_ready())
	{
		std::printf("[AIMER_VISUAL] [ERROR] OpenVINO inference is not ready\n");
		return 7;
	}

	app::auto_aim::Detector detector(detector_config, std::move(inference));
	app::auto_aim::Solver solver(solver_config);
	app::auto_aim::Tracker tracker(tracker_config);
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));
	app::auto_aim::Aimer aimer(aimer_config);

	if(!auto_aim.is_ready())
	{
		std::printf("[AIMER_VISUAL] [ERROR] AutoAim is not ready\n");
		return 8;
	}

	cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

	cv::Mat frame;
	std::size_t frame_index = 0;
	bool paused = false;

	auto run_one_frame = [&]() -> bool {
		if(!video.read(frame) || frame.empty())
		{
			return false;
		}

		app::auto_aim::FrameContext context;
		context.image = frame;

		Eigen::Quaterniond q = Eigen::Quaterniond::Identity();

		if(has_sync_quaternions)
		{
			if(frame_index >= quaternions.size())
			{
				return false;
			}

			context.timestamp_s = quaternions[frame_index].timestamp_s;
			q = quaternions[frame_index].q;
			context.q_imu_body_to_world = q;
		}
		else
		{
			context.timestamp_s = static_cast<double>(frame_index) / fps;
		}

		app::auto_aim::AutoAimDebugData debug;
		const auto result = auto_aim.process(context, &debug);

		cv::Mat canvas = frame.clone();

		// 1. Detector 装甲板轮廓。
		for(std::size_t i = 0; i < debug.detected_armors.size(); ++i)
		{
			const auto& armor = debug.detected_armors[i];

			if(armor.points.size() == 4)
			{
				tools::draw_points(canvas, armor.points, kDetectedArmorColor, 2);
			}
		}

		const cv::Point origin(10, 30);

		if(result.tracked_target.has_value())
		{
			const app::auto_aim::TrackedTarget& tracked = *result.tracked_target;

			// 2. 车辆中心 + 速度向量。
			const auto center_pixel = project_world_to_image(tracked.center_in_world, solver_config, q);

			if(center_pixel.has_value())
			{
				const cv::Point center(static_cast<int>(center_pixel->x),
				                       static_cast<int>(center_pixel->y));

				cv::circle(canvas, center, 5, kCenterColor, -1);
				draw_cross(canvas, center, kCenterColor, 8);

				const double velocity_scale = 0.5;
				const Eigen::Vector3d velocity_tip =
				    tracked.center_in_world + velocity_scale * tracked.velocity_in_world;

				const auto tip_pixel = project_world_to_image(velocity_tip, solver_config, q);

				if(tip_pixel.has_value())
				{
					const cv::Point tip(static_cast<int>(tip_pixel->x),
					                    static_cast<int>(tip_pixel->y));
					cv::arrowedLine(canvas, center, tip, kVelocityColor, 2);
				}
			}

			// 3. 预测 armor hypothesis。
			for(const auto& hypothesis: tracked.predicted_armors)
			{
				const auto pixel = project_world_to_image(hypothesis.position_in_world, solver_config, q);

				if(!pixel.has_value())
				{
					continue;
				}

				const cv::Point center(static_cast<int>(pixel->x), static_cast<int>(pixel->y));
				cv::circle(canvas, center, 4, kHypothesisColor, 1);

				char label[16];
				std::snprintf(label, sizeof(label), "%d", hypothesis.armor_id);
				cv::putText(canvas, label, cv::Point(center.x + 6, center.y),
				            cv::FONT_HERSHEY_SIMPLEX, 0.5, kHypothesisColor, 1);
			}

			// 4. Aimer 求解。
			const bool aimable = tracked.state == app::auto_aim::TrackerState::Tracking
			    || tracked.state == app::auto_aim::TrackerState::TempLost;

			if(aimable)
			{
				app::auto_aim::AimerDebugData aimer_debug;
				const auto solution =
				    aimer.aim(tracked, context.timestamp_s, bullet_speed_mps, &aimer_debug);

				LOG_DEBUG("AIMER_VISUAL",
				          "frame {}: status={} yaw={:.3f} pitch={:.3f} selected={} fire={} "
				          "aim=({:.3f},{:.3f},{:.3f}) flight={:.3f}s",
				          frame_index + 1, aim_status_name(solution.status), solution.yaw_rad,
				          solution.pitch_rad,
				          solution.selected_armor_id.has_value()
				              ? std::to_string(*solution.selected_armor_id)
				              : std::string("none"),
				          solution.fire_allowed ? 1 : 0, aimer_debug.aim_point_in_world.x(),
				          aimer_debug.aim_point_in_world.y(), aimer_debug.aim_point_in_world.z(),
				          aimer_debug.flight_time_s);

				// 选中装甲板高亮为红色粗圈。
				if(solution.selected_armor_id.has_value())
				{
					const int selected = *solution.selected_armor_id;

					for(const auto& hypothesis: tracked.predicted_armors)
					{
						if(hypothesis.armor_id != selected)
						{
							continue;
						}

						const auto pixel =
						    project_world_to_image(hypothesis.position_in_world, solver_config, q);

						if(pixel.has_value())
						{
							const cv::Point center(static_cast<int>(pixel->x),
							                       static_cast<int>(pixel->y));
							cv::circle(canvas, center, 8, kSelectedColor, 3);
						}
					}
				}

				// 瞄准点（黄色十字 + 圆圈），仅在有效解时绘制。
				if(solution.valid)
				{
					const auto aim_pixel =
					    project_world_to_image(aimer_debug.aim_point_in_world, solver_config, q);

					if(aim_pixel.has_value())
					{
						const cv::Point aim(static_cast<int>(aim_pixel->x),
						                    static_cast<int>(aim_pixel->y));
						draw_cross(canvas, aim, kAimPointColor, 12, 3);
						cv::circle(canvas, aim, 10, kAimPointColor, 2);
					}
				}

				// 状态文本。
				char status_line[256];

				if(solution.valid)
				{
					std::snprintf(status_line, sizeof(status_line),
					              "aimer: %s  yaw=%.3frad(%.1fdeg)  pitch=%.3frad(%.1fdeg)",
					              aim_status_name(solution.status), solution.yaw_rad,
					              solution.yaw_rad * kRadToDeg, solution.pitch_rad,
					              solution.pitch_rad * kRadToDeg);
				}
				else
				{
					std::snprintf(status_line, sizeof(status_line), "aimer: %s (invalid)",
					              aim_status_name(solution.status));
				}

				cv::putText(canvas, status_line, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6,
				            solution.valid ? kTextColor : cv::Scalar(0, 0, 255), 1);

				char select_line[128];
				std::snprintf(select_line, sizeof(select_line),
				              "selected_armor=%s  fire_allowed=%d",
				              solution.selected_armor_id.has_value()
				                  ? std::to_string(*solution.selected_armor_id).c_str()
				                  : "none",
				              solution.fire_allowed ? 1 : 0);
				cv::putText(canvas, select_line, cv::Point(origin.x, origin.y + 22),
				            cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1);

				char ball_line[256];
				std::snprintf(ball_line, sizeof(ball_line),
				              "aim_point=(%.3f,%.3f,%.3f)  flight=%.3fs  converged=%d  iters=%d",
				              aimer_debug.aim_point_in_world.x(), aimer_debug.aim_point_in_world.y(),
				              aimer_debug.aim_point_in_world.z(), aimer_debug.flight_time_s,
				              aimer_debug.ballistic_converged ? 1 : 0,
				              aimer_debug.refinement_iterations);
				cv::putText(canvas, ball_line, cv::Point(origin.x, origin.y + 44),
				            cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1);
			}
			else
			{
				char wait_line[128];
				std::snprintf(wait_line, sizeof(wait_line), "aimer: waiting (tracker %s)",
				              tracker_state_name(tracked.state));
				cv::putText(canvas, wait_line, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6,
				            cv::Scalar(0, 200, 255), 1);
			}

			// Tracker 状态行（放在 Aimer 状态下方）。
			const int tracker_y = origin.y + 88;

			char tracker_line[128];
			std::snprintf(tracker_line, sizeof(tracker_line), "tracker: %s  has_meas=%d",
			              tracker_state_name(tracked.state), tracked.has_measurement ? 1 : 0);
			cv::putText(canvas, tracker_line, cv::Point(origin.x, tracker_y),
			            cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1);

			char pose_line[256];
			std::snprintf(pose_line, sizeof(pose_line), "center=%.3f,%.3f,%.3f  vel=%.3f,%.3f,%.3f",
			              tracked.center_in_world.x(), tracked.center_in_world.y(),
			              tracked.center_in_world.z(), tracked.velocity_in_world.x(),
			              tracked.velocity_in_world.y(), tracked.velocity_in_world.z());
			cv::putText(canvas, pose_line, cv::Point(origin.x, tracker_y + 22),
			            cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1);
		}
		else
		{
			cv::putText(canvas, "tracker: Lost (no target)", cv::Point(10, 30),
			            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			cv::putText(canvas, "aimer: no valid target", cv::Point(10, 60),
			            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
		}

		cv::imshow(kWindowName, canvas);

		++frame_index;
		return true;
	};

	while(true)
	{
		if(!paused)
		{
			if(!run_one_frame())
			{
				break;
			}
		}

		int wait_ms = paused ? 20 : std::max(1, static_cast<int>(std::round(1000.0 / fps)));
		const int key = cv::waitKey(wait_ms);

		if(key == 'q' || key == 'Q' || key == 27)
		{
			break;
		}

		if(key == ' ')
		{
			paused = !paused;
			continue;
		}

		if(paused && (key == 'n' || key == 'N'))
		{
			if(!run_one_frame())
			{
				break;
			}
		}
	}

	video.release();
	cv::destroyAllWindows();

	std::printf("[AIMER_VISUAL] processed %zu frames\n", frame_index);

	tools::Logger::instance().shutdown();
	return 0;
}
