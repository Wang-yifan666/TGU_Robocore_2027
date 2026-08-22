/**
 * @file test_auto_aim_visual.cpp
 * @brief AutoAim 离线视觉回放调试工具（第一阶段，人工交互，不注册 CTest）。
 *
 * 数据流：
 *   video(--video) + 可选同步 quaternion(--quaternion, timestamp w x y z)
 *     -> FrameContext
 *     -> AutoAim::process(frame, &debug)
 *     -> AimResult + AutoAimDebugData
 *     -> task 层 overlay -> cv::imshow
 *
 * 显示内容：
 * - Detector 全部 Armor（四点闭合轮廓 + name/type/color/confidence）
 * - 最终被选中且 PnP 成功的 Armor（明显区分，绿色粗轮廓）
 * - Solver 信息：xyz_in_gimbal / distance
 * - 有同步 quaternion 时显示 world(sync input)，否则显示 world: N/A
 * - 帧状态：frame index / detection count / AimState / has_target /
 *   inference time / postprocess time / total processing time
 *
 * 回放控制：
 * - Space      暂停 / 继续
 * - N / Right  暂停状态下单帧前进
 * - Q / ESC    退出
 *
 * 播放节奏：waitKey 延迟 = max(1, 1000/fps - 本帧 process 耗时)，
 * 避免处理完成后再次等待完整帧周期造成慢放。
 *
 * 注意：
 * - 本工具不改变 AutoAim 算法行为，debug 仅旁路观察；
 * - 不二次推理、不对所有 Armor 额外调用 Solver；
 * - world 坐标仅在存在同步 quaternion 时显示并标注 sync input，
 *   且不声称 world frame 已 validated。
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"
#include "tools/img_tools.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

	constexpr const char* MODULE = "AUTO_AIM_VISUAL";
	constexpr const char* kWindowName = "auto_aim_visual";

	constexpr std::size_t kArmorCornerCount = 4;

	// BGR 颜色常量。
	const cv::Scalar kNormalArmorColor{255, 200, 0};  // 青
	const cv::Scalar kSelectedArmorColor{0, 255, 0};   // 绿
	const cv::Scalar kTextColor{255, 255, 255};
	const cv::Scalar kInfoColor{80, 200, 255};
	const cv::Scalar kWarnColor{0, 0, 255};

	using SteadyClock = std::chrono::steady_clock;

	struct SyncQuaternion
	{
		double timestamp_s = 0.0;
		Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
	};

	double elapsed_ms(const SteadyClock::time_point& begin, const SteadyClock::time_point& end)
	{
		return std::chrono::duration<double, std::milli>(end - begin).count();
	}

	/**
	 * @brief 尝试加载与视频同步的 quaternion 文件（timestamp w x y z）。
	 * @return true 成功加载；false 文件不存在/格式非法。
	 */
	bool try_load_quaternions(const std::string& path, std::vector<SyncQuaternion>& out)
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
			if(!std::isfinite(timestamp) || !std::isfinite(w) || !std::isfinite(x)
			   || !std::isfinite(y) || !std::isfinite(z))
			{
				std::printf("[%s] [WARN] non-finite quaternion fields at line %zu\n", MODULE,
				            count + 1);
				return false;
			}

			const double norm_squared = w * w + x * x + y * y + z * z;

			if(norm_squared <= 1e-12)
			{
				std::printf("[%s] [WARN] zero-norm quaternion at line %zu\n", MODULE, count + 1);
				return false;
			}

			SyncQuaternion sync;
			sync.timestamp_s = timestamp;
			sync.q = Eigen::Quaterniond(w, x, y, z);
			sync.q.normalize();

			out.emplace_back(std::move(sync));
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

	std::string state_string(app::auto_aim::AimState state)
	{
		switch(state)
		{
		case app::auto_aim::AimState::Idle:
			return "Idle";
		case app::auto_aim::AimState::NoFrame:
			return "NoFrame";
		case app::auto_aim::AimState::NoTarget:
			return "NoTarget";
		case app::auto_aim::AimState::Detecting:
			return "Detecting";
		case app::auto_aim::AimState::Tracking:
			return "Tracking";
		case app::auto_aim::AimState::TargetLocked:
			return "TargetLocked";
		case app::auto_aim::AimState::Error:
		default:
			return "Error";
		}
	}

	/**
	 * @brief 绘制单块装甲板：四点闭合轮廓（tools::draw_points 自动闭合）。
	 */
	void draw_armor_contour(cv::Mat& canvas, const app::auto_aim::Armor& armor,
	                        const cv::Scalar& color, int thickness)
	{
		if(armor.points.size() != kArmorCornerCount)
		{
			return;
		}

		tools::draw_points(canvas, armor.points, color, thickness);
	}

	void draw_status_bar(cv::Mat& canvas, std::size_t frame_index,
	                     const app::auto_aim::AutoAimDebugData& debug,
	                     const app::auto_aim::AimResult& result, double processing_ms,
	                     bool has_sync_quaternions)
	{
		const int bar_height = 130;
		const cv::Rect bar(0, canvas.rows - bar_height, canvas.cols, bar_height);
		cv::rectangle(canvas, bar, cv::Scalar(0, 0, 0), cv::FILLED);

		const cv::Point origin(10, canvas.rows - bar_height + 22);

		// 第一行：帧信息。
		char line0[256];
		std::snprintf(line0, sizeof(line0), "frame %zu | detections %zu | state=%s has_target=%d",
		              frame_index, debug.detected_armors.size(),
		              state_string(result.state).c_str(), static_cast<int>(result.has_target));
		tools::draw_text(canvas, line0, origin, kTextColor, 0.6, 1);

		// 第二行：耗时。
		char line1[256];
		std::snprintf(line1, sizeof(line1), "inference %.2f ms | postprocess %.2f ms | total %.2f ms",
		              debug.inference_time_ms, debug.postprocess_time_ms, processing_ms);
		tools::draw_text(canvas, line1, cv::Point(origin.x, origin.y + 22), kTextColor, 0.6, 1);

		// 第三行：world 状态提示。
		const std::string world_hint =
		    has_sync_quaternions ? "world: sync input (not validated)"
		                         : "world: N/A (no synchronized quaternion)";
		tools::draw_text(canvas, world_hint, cv::Point(origin.x, origin.y + 44),
		                 has_sync_quaternions ? kInfoColor : kWarnColor, 0.6, 1);

		// 第四行：操作提示。
		tools::draw_text(canvas, "Space pause/resume | N / Right step | Q / ESC quit",
		                 cv::Point(origin.x, origin.y + 66), cv::Scalar(255, 255, 255), 0.5, 1);
	}

	void draw_solver_info(cv::Mat& canvas, const app::auto_aim::AimResult& result,
	                      bool has_sync_quaternions)
	{
		if(!result.has_target)
		{
			return;
		}

		const cv::Point origin(10, 30);

		char line0[256];
		std::snprintf(line0, sizeof(line0), "target: name=%s type=%s color=%s conf=%.3f",
		              std::string(app::auto_aim::to_string(result.target.name)).c_str(),
		              std::string(app::auto_aim::to_string(result.target.type)).c_str(),
		              std::string(app::auto_aim::to_string(result.target.color)).c_str(),
		              result.target.confidence);
		tools::draw_text(canvas, line0, origin, kInfoColor, 0.6, 1);

		char line1[256];
		std::snprintf(line1, sizeof(line1),
		              "xyz_in_gimbal=(%.3f, %.3f, %.3f) m  distance=%.3f m",
		              result.target.xyz_in_gimbal.x(), result.target.xyz_in_gimbal.y(),
		              result.target.xyz_in_gimbal.z(), result.distance);
		tools::draw_text(canvas, line1, cv::Point(origin.x, origin.y + 22), kTextColor, 0.6, 1);

		if(has_sync_quaternions)
		{
			char line2[256];
			std::snprintf(line2, sizeof(line2), "world(sync input)=(%.3f, %.3f, %.3f) m",
			              result.target.xyz_in_world.x(), result.target.xyz_in_world.y(),
			              result.target.xyz_in_world.z());
			tools::draw_text(canvas, line2, cv::Point(origin.x, origin.y + 44), kTextColor, 0.6,
			                 1);
		}
		else
		{
			tools::draw_text(canvas, "world: N/A",
			                 cv::Point(origin.x, origin.y + 44), kWarnColor, 0.6, 1);
		}
	}

	bool is_n_or_right_arrow(int key)
	{
		// 'n' / 'N' 以及常见的 Right Arrow 键码（Linux/X11 0xff53，Windows 0x01100013）。
		return key == 'n' || key == 'N' || key == 65363 || key == 1113939;
	}

	// OpenCV CommandLineParser 仅支持 "--opt=value" 语法；"--opt value" 会被
	// 当作布尔开关。这里把常见取值型选项的两段形式合并为等号形式，便于日常使用。
	std::vector<std::string> normalize_args(int argc, char** argv)
	{
		const std::vector<std::string> value_options = {
		    "--video", "--quaternion", "--start-frame", "--end-frame"};

		std::vector<std::string> normalized;
		normalized.reserve(static_cast<std::size_t>(argc));

		for(int index = 0; index < argc; ++index)
		{
			const std::string current = argv[index];

			const bool is_value_option =
			    std::find(value_options.begin(), value_options.end(), current) != value_options.end();

			normalized.push_back(current);

			// 当前参数是取值型选项，且下一个参数存在且不以 '--' 开头 -> 合并。
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
	const std::vector<std::string> normalized_args = normalize_args(argc, argv);

	std::vector<char*> arg_ptrs;
	arg_ptrs.reserve(normalized_args.size());

	for(const auto& arg: normalized_args)
	{
		arg_ptrs.push_back(const_cast<char*>(arg.c_str()));
	}

	const cv::String keys =
	    "{help h usage ? |           | print this help}"
	    "{video          |           | input video path (default data/demo/demo.avi)}"
	    "{quaternion     |           | optional synchronized quaternion file "
	    "timestamp w x y z (default data/demo/demo.txt)}"
	    "{start-frame    | 0         | first frame index to process}"
	    "{end-frame      | 0         | last frame index, inclusive; 0 = no limit}";

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

	const int start_frame = parser.get<int>("start-frame");
	const int end_frame = parser.get<int>("end-frame");

	if(!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	// ============================================================
	// 1. demo 视频与 quaternion 数据
	// ============================================================
	cv::VideoCapture video(video_path);

	if(!video.isOpened())
	{
		std::printf("[%s] [ERROR] failed to open video: %s\n", MODULE, video_path.c_str());
		return 2;
	}

	if(start_frame > 0)
	{
		video.set(cv::CAP_PROP_POS_FRAMES, start_frame);
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
		std::printf("[%s] [WARN] No synchronized quaternion data found at %s.\n"
		            "[%s] [WARN] World-frame results are not displayed.\n",
		            MODULE, quaternion_path.c_str(), MODULE);
	}

	// ============================================================
	// 2. 配置加载（demo 使用 demo-only solver 标定）
	// ============================================================
	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";

	app::auto_aim::DetectorConfig detector_config;

	if(!app::auto_aim::load_detector_config(detector_config_path, detector_config))
	{
		std::printf("[%s] [ERROR] failed to load detector config: %s\n", MODULE,
		            detector_config_path.c_str());
		return 3;
	}

	app::auto_aim::SolverConfig solver_config;

	if(!app::auto_aim::load_solver_config(solver_config_path, solver_config))
	{
		std::printf("[%s] [ERROR] failed to load solver config: %s\n", MODULE,
		            solver_config_path.c_str());
		return 4;
	}

	// ============================================================
	// 3. 构建真实 OpenVINO 推理后端 + Detector + Solver + AutoAim
	// ============================================================
	auto inference = std::make_unique<app::auto_aim::OpenVINOInference>(
	    detector_config.model_path, detector_config.device,
	    detector_config.inference_score_threshold);

	if(!inference->is_ready())
	{
		std::printf("[%s] [ERROR] OpenVINO inference is not ready\n", MODULE);
		return 5;
	}

	app::auto_aim::Detector detector(detector_config, std::move(inference));
	app::auto_aim::Solver solver(solver_config);
	app::auto_aim::Tracker tracker(app::auto_aim::make_default_tracker_config());
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));

	if(!auto_aim.is_ready())
	{
		std::printf("[%s] [ERROR] AutoAim dependencies are not ready\n", MODULE);
		return 6;
	}

	// ============================================================
	// 4. 回放主循环
	// ============================================================
	cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

	cv::Mat frame;
	std::size_t frame_index = static_cast<std::size_t>(start_frame);
	std::size_t processed_frames = 0;

	bool paused = false;
	double last_processing_ms = 0.0;

	auto run_one_frame = [&]() -> bool {
		if(end_frame > 0 && static_cast<int>(frame_index) > end_frame)
		{
			return false;
		}

		if(!video.read(frame))
		{
			return false;
		}

		if(frame.empty())
		{
			return false;
		}

		app::auto_aim::FrameContext context;
		context.image = frame;

		if(has_sync_quaternions)
		{
			if(frame_index >= quaternions.size())
			{
				// 同步数据提前耗尽：停止处理未同步的视频尾部，
				// 不复用最后一个 quaternion、不回退 identity。
				std::printf(
				    "[%s] [WARN] sync truncated at frame %zu (quaternion lines=%zu); "
				    "stopping video tail\n",
				    MODULE, frame_index, quaternions.size());
				return false;
			}

			const auto& sync = quaternions[frame_index];
			context.timestamp_s = sync.timestamp_s;
			context.q_imu_body_to_world = sync.q;
		}
		else
		{
			context.timestamp_s = static_cast<double>(frame_index) / fps;
		}

		// 单帧处理 + 旁路 debug。
		app::auto_aim::AutoAimDebugData debug;

		const auto process_begin = SteadyClock::now();
		const auto result = auto_aim.process(context, &debug);
		const auto process_end = SteadyClock::now();

		last_processing_ms = elapsed_ms(process_begin, process_end);

		// overlay 在 task 层完成；不修改输入 frame，使用 clone。
		cv::Mat canvas = frame.clone();

		// 所有 Detector 输出装甲板 + 基本识别信息。
		for(std::size_t index = 0; index < debug.detected_armors.size(); ++index)
		{
			const auto& armor = debug.detected_armors[index];
			const bool is_selected = debug.selected_armor_index.has_value()
			    && *debug.selected_armor_index == index;

			const cv::Scalar contour_color =
			    is_selected ? kSelectedArmorColor : kNormalArmorColor;
			const int contour_thickness = is_selected ? 4 : 2;

			draw_armor_contour(canvas, armor, contour_color, contour_thickness);

			if(is_selected)
			{
				tools::draw_point(canvas, cv::Point(static_cast<int>(armor.center.x),
				                                    static_cast<int>(armor.center.y)),
				                  kSelectedArmorColor, 4);
			}

			// 识别信息放在装甲板左上角附近。
			char label[128];
			std::snprintf(label, sizeof(label), "%s %s %s %.2f",
			              std::string(app::auto_aim::to_string(armor.name)).c_str(),
			              std::string(app::auto_aim::to_string(armor.type)).c_str(),
			              std::string(app::auto_aim::to_string(armor.color)).c_str(),
			              armor.confidence);

			const cv::Point label_origin(static_cast<int>(armor.center.x) - 30,
			                             static_cast<int>(armor.center.y) + 30);
			tools::draw_text(canvas, label, label_origin, is_selected ? kSelectedArmorColor
			                                                          : kTextColor,
			                 0.5, 1);
		}

		// Solver 信息仅针对最终成功目标绘制。
		draw_solver_info(canvas, result, has_sync_quaternions);

		// 状态栏。
		draw_status_bar(canvas, frame_index, debug, result, last_processing_ms,
		                has_sync_quaternions);

		cv::imshow(kWindowName, canvas);

		++processed_frames;
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

		// 播放节奏：waitKey 延迟 = max(1, 帧周期 - 本帧处理耗时)。
		int wait_ms = 20;

		if(!paused)
		{
			const int frame_period_ms =
			    std::max(1, static_cast<int>(std::round(1000.0 / fps)));
			const int processing_ms = static_cast<int>(std::round(last_processing_ms));
			wait_ms = std::max(1, frame_period_ms - processing_ms);
		}

		const int key = cv::waitKey(wait_ms);

		if(key == 'q' || key == 'Q' || key == 27) // 27 = ESC
		{
			break;
		}

		if(key == ' ')
		{
			paused = !paused;
			continue;
		}

		if(paused && is_n_or_right_arrow(key))
		{
			// 暂停状态下单帧前进。
			if(!run_one_frame())
			{
				break;
			}
		}
	}

	video.release();
	cv::destroyAllWindows();

	std::printf("\n================= Visual Replay Summary =================\n");
	std::printf("video:              %s\n", video_path.c_str());
	std::printf("processed_frames:   %zu\n", processed_frames);
	std::printf("start_frame:        %d\n", start_frame);
	std::printf("end_frame:          %s\n",
	            end_frame > 0 ? std::to_string(end_frame).c_str() : "unlimited");
	std::printf("sync_quaternion:    %s\n",
	            has_sync_quaternions ? "yes" : "no (world: N/A)");
	std::printf("========================================================\n");

	return 0;
}