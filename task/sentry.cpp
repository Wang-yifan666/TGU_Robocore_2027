//
// Created by tgu on 2026/4/14.
//

#include <csignal>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/core.hpp>

#include "app/auto_aim/auto_aim.hpp"
#include "tools/logger.hpp"

namespace
{
	static constexpr const char* MODULE = "SENTRY";
	volatile std::sig_atomic_t g_running = 1;

	void signal_handler(int)
	{
		g_running = 0;
	}

	struct RuntimeConfig
	{
		bool use_camera = false;
		bool use_serial = false;
		bool use_foxglove = false;

		int loop_period_ms = 10;
	};

	double now_seconds()
	{
		using clock = std::chrono::steady_clock;
		return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
	}

	RuntimeConfig load_runtime_config()
	{
		RuntimeConfig config;

		// TODO:
		// 后续从 config/task/sentry.toml 或 config/app/auto_aim/*.toml 读取

		return config;
	}

	bool read_frame_stub(cv::Mat& image)
	{
		// TODO:
		// 后续替换为 io::HikRobotCamera::read(image)
		image = cv::Mat::zeros(480, 640, CV_8UC3);
		return true;
	}

} // namespace

int main(int argc, char** argv)
{
	(void)argc;
	(void)argv;

	std::signal(SIGINT, signal_handler);
	std::signal(SIGTERM, signal_handler);

	std::filesystem::create_directories(std::string(PROJECT_SOURCE_DIR) + "/data/logs");

	tools::LoggerConfig logger_config;
	logger_config.level = tools::LogLevel::Debug;
	logger_config.enable_console = true;
	logger_config.enable_file = true;
	logger_config.file_path = std::string(PROJECT_SOURCE_DIR) + "/data/logs/sentry.log";

	if(!tools::Logger::instance().init(logger_config))
	{
		std::cerr << "[SENTRY] Logger initialization failed, log will not be written to file\n";
	}


	LOG_INFO(MODULE, "sentry task starting");

	RuntimeConfig runtime_config = load_runtime_config();

	// 生产入口暂未装配真实 detector / solver 依赖。
	// 禁止使用 demo-only solver 标定（config/app/auto_aim/solver_demo.toml）。
	// 待生产 camera intrinsic / distortion / hand-eye calibration 到位后，
	// 由 task 层加载配置并注入 Detector / Solver。
	app::auto_aim::DetectorConfig detector_config;
	app::auto_aim::Detector detector(detector_config, nullptr);

	app::auto_aim::SolverConfig solver_config;
	app::auto_aim::Solver solver(solver_config);

	app::auto_aim::Tracker tracker(app::auto_aim::make_default_tracker_config());
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));

	// 依赖未就绪时直接退出，避免每 10ms 空转刷 ERROR。
	if(!auto_aim.is_ready())
	{
		LOG_ERROR(MODULE, "auto aim dependencies not ready: detector/solver not configured");
		return -1;
	}

	while(g_running)
	{
		cv::Mat image;

		if(!read_frame_stub(image))
		{
			LOG_WARN(MODULE, "failed to read frame");
			std::this_thread::sleep_for(std::chrono::milliseconds(runtime_config.loop_period_ms));
			continue;
		}

		app::auto_aim::FrameContext frame_context;
		frame_context.image = image;
		frame_context.timestamp_s = now_seconds();

		auto result = auto_aim.process(frame_context);

		if(result.has_target)
		{
			// pre-tracker 阶段：仅输出 raw geometric observation，
			// 不把 yaw/pitch 当作最终云台控制命令。
			LOG_INFO(MODULE, "target detected/solved, xyz_in_gimbal=({:.3f}, {:.3f}, {:.3f}) m, "
			                 "distance={:.3f} m",
			         result.target.xyz_in_gimbal.x(), result.target.xyz_in_gimbal.y(),
			         result.target.xyz_in_gimbal.z(), result.distance);

			// TODO: 串口发送给下位机
		}
		else
		{
			LOG_DEBUG(MODULE, "no target");
		}

		// TODO: 接入 Foxglove / debug image / performance profiler

		std::this_thread::sleep_for(std::chrono::milliseconds(runtime_config.loop_period_ms));
	}

	LOG_INFO(MODULE, "sentry task stopped");

	return 0;
}
