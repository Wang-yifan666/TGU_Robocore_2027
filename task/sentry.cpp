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

#include "app/auto_aim/aimer_config.hpp"
#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/shooter_config.hpp"
#include "app/auto_aim/tracker_config.hpp"
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

	app::auto_aim::TrackerConfig tracker_config;
	app::auto_aim::AimerConfig aimer_config;
	app::auto_aim::ShooterConfig shooter_config;

	if(!app::auto_aim::load_tracker_config(
	       std::string(PROJECT_SOURCE_DIR) + "/config/app/auto_aim/tracker.toml",
	       tracker_config))
	{
		LOG_ERROR(MODULE, "failed to load tracker config");
		return -1;
	}

	if(!app::auto_aim::load_aimer_config(
	       std::string(PROJECT_SOURCE_DIR) + "/config/app/auto_aim/aimer.toml",
	       aimer_config))
	{
		LOG_ERROR(MODULE, "failed to load aimer config");
		return -1;
	}

	if(!app::auto_aim::load_shooter_config(
	       std::string(PROJECT_SOURCE_DIR) + "/config/app/auto_aim/shooter.toml",
	       shooter_config))
	{
		LOG_ERROR(MODULE, "failed to load shooter config");
		return -1;
	}

	app::auto_aim::Tracker tracker(tracker_config);
	app::auto_aim::Aimer aimer(aimer_config);
	app::auto_aim::Shooter shooter(shooter_config);
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker),
	                                std::move(aimer), std::move(shooter));

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

		// TODO(task/io): 接入 cboard 弹速与云台 yaw 反馈；未接入前保持 NaN，
		// pipeline 将 fail-safe（fire=false）。禁止填入伪造的 0.0。

		auto result = auto_aim.process(frame_context);

		// 只有 Tracking / TempLost 才是 confirmed 控制目标（has_target == true）。
		// Detecting 阶段即使 tracked_target 存在也只是未确认候选，不得作为控制目标。
		if(result.has_target && result.tracked_target.has_value())
		{
			const auto& tracked = *result.tracked_target;

			LOG_INFO(MODULE, "confirmed target center=({:.3f}, {:.3f}, {:.3f}) m, "
			                 "vel=({:.3f}, {:.3f}, {:.3f}) m/s, yaw={:.3f} rad, has_meas={}",
			         tracked.center_in_world.x(), tracked.center_in_world.y(),
			         tracked.center_in_world.z(), tracked.velocity_in_world.x(),
			         tracked.velocity_in_world.y(), tracked.velocity_in_world.z(), tracked.yaw,
			         tracked.has_measurement ? 1 : 0);

			// TODO: 串口发送给下位机（未来 Aimer/Planner 消费 tracked）
		}
		else if(result.tracked_target.has_value())
		{
			// Detecting：tracked_target 存在但未确认，不作为控制目标。
			const auto& tracked = *result.tracked_target;
			LOG_DEBUG(MODULE, "unconfirmed target (Detecting), center=({:.3f}, {:.3f}, {:.3f}) m",
			          tracked.center_in_world.x(), tracked.center_in_world.y(),
			          tracked.center_in_world.z());
		}
		else
		{
			LOG_DEBUG(MODULE, "no tracked target (Lost)");
		}

		// TODO: 接入 Foxglove / debug image / performance profiler

		std::this_thread::sleep_for(std::chrono::milliseconds(runtime_config.loop_period_ms));
	}

	LOG_INFO(MODULE, "sentry task stopped");

	return 0;
}
