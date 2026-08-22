/**
 * @file test_tracker_replay.cpp
 * @brief Tracker v1 真实/录制数据 replay + 结构化诊断 + quantitative validation。
 *
 * 这是 manual integration/replay executable，不注册进默认 CTest。
 *
 * 数据流：
 *   data/demo/demo.avi (+ demo.txt 同步 quaternion)
 *     -> OpenVINOInference -> Detector -> Solver(全部 detection)
 *     -> AutoAim(=Detector+Solver+Tracker)
 *     -> AimResult::tracked_target
 *     -> 逐帧 CSV trace + 定量 summary
 *
 * 输出：
 * - data/logs/tracker_replay.csv   （每帧一行结构化诊断）
 * - stdout quantitative summary
 *
 * 注意：
 * - 无 ground truth，因此不声称 absolute tracking accuracy；
 *   仅验证 continuity / consistency / numerical health。
 * - Foxglove transport 尚未接入；CSV 字段构成后续 adapter 的 stable schema。
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace
{

	constexpr const char* kModule = "TRACKER_REPLAY";

	struct ReplayStats
	{
		std::size_t frames = 0;
		std::size_t frames_with_observations = 0;

		std::size_t successful_associations = 0;
		std::size_t rejected_associations = 0;

		std::size_t successful_corrections = 0;
		std::size_t correction_failures = 0;

		std::size_t tracker_initializations = 0;
		std::size_t lost_transitions = 0;
		std::size_t temp_lost_frames = 0;
		std::size_t reacquisitions = 0;

		std::size_t armor_switch_count = 0;

		// board-switch continuity：matched armor_id 变化时，
		// 记录 correction 后 center 相对上一帧 center 的跳变量。
		std::vector<double> center_jumps_m;
	};

	bool is_finite_vector3(const Eigen::Vector3d& v)
	{
		return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
	}

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

		while(file >> timestamp >> w >> x >> y >> z)
		{
			const double norm2 = w * w + x * x + y * y + z * z;

			if(!std::isfinite(timestamp) || norm2 <= 1e-12)
			{
				return false;
			}

			Eigen::Quaterniond q(w, x, y, z);
			q.normalize();
			out.emplace_back(timestamp, q);
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

	void write_csv_header(std::ofstream& out)
	{
		out << "timestamp_s,state,has_measurement,name,type,color,"
		    << "center_x,center_y,center_z,"
		    << "vel_x,vel_y,vel_z,"
		    << "yaw,yaw_rate,radius,delta_radius,delta_z,"
		    << "matched_observation_index,matched_armor_id,"
		    << "innovation_x,innovation_y,innovation_z,innovation_yaw,nis,"
		    << "predicted_armors\n";
	}

	void write_csv_row(std::ofstream& out, const app::auto_aim::TrackedTarget& t)
	{
		out << t.timestamp_s << ','
		    << tracker_state_name(t.state) << ','
		    << (t.has_measurement ? 1 : 0) << ','
		    << std::string(app::auto_aim::to_string(t.name)).c_str() << ','
		    << std::string(app::auto_aim::to_string(t.type)).c_str() << ','
		    << std::string(app::auto_aim::to_string(t.color)).c_str() << ','
		    << t.center_in_world.x() << ','
		    << t.center_in_world.y() << ','
		    << t.center_in_world.z() << ','
		    << t.velocity_in_world.x() << ','
		    << t.velocity_in_world.y() << ','
		    << t.velocity_in_world.z() << ','
		    << t.yaw << ','
		    << t.yaw_rate << ','
		    << t.radius << ','
		    << t.delta_radius << ','
		    << t.delta_z << ',';

		if(t.matched_observation_index.has_value())
		{
			out << *t.matched_observation_index << ',';
		}
		else
		{
			out << ',';
		}

		if(t.matched_armor_id.has_value())
		{
			out << *t.matched_armor_id << ',';
		}
		else
		{
			out << ',';
		}

		if(t.innovation.has_value() && t.innovation->size() == 4)
		{
			out << (*t.innovation)(0) << ','
			    << (*t.innovation)(1) << ','
			    << (*t.innovation)(2) << ','
			    << (*t.innovation)(3) << ',';
		}
		else
		{
			out << ",,,,";
		}

		if(t.nis.has_value())
		{
			out << *t.nis;
		}

		out << ',' << t.predicted_armors.size() << '\n';
	}

	double percentile(const std::vector<double>& values, double p)
	{
		if(values.empty())
		{
			return std::numeric_limits<double>::quiet_NaN();
		}

		auto sorted = values;
		std::sort(sorted.begin(), sorted.end());

		const double rank = p * static_cast<double>(sorted.size() - 1);
		const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
		const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));

		if(lo == hi)
		{
			return sorted[lo];
		}

		const double frac = rank - static_cast<double>(lo);
		return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
	}

} // namespace

int main()
{
	const std::string project_root = std::string(PROJECT_SOURCE_DIR) + "/";

	const std::string video_path = project_root + "data/demo/demo.avi";
	const std::string quaternion_path = project_root + "data/demo/demo.txt";
	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";
	const std::string csv_path = project_root + "data/logs/tracker_replay.csv";

	std::filesystem::create_directories(project_root + "data/logs");

	cv::VideoCapture video(video_path);

	if(!video.isOpened())
	{
		std::printf("[%s] [ERROR] failed to open video: %s\n", kModule, video_path.c_str());
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
		std::printf("[%s] [WARN] no synchronized quaternion data; using identity\n", kModule);
	}

	app::auto_aim::DetectorConfig detector_config;
	if(!app::auto_aim::load_detector_config(detector_config_path, detector_config))
	{
		std::printf("[%s] [ERROR] failed to load detector config\n", kModule);
		return 2;
	}

	app::auto_aim::SolverConfig solver_config;
	if(!app::auto_aim::load_solver_config(solver_config_path, solver_config))
	{
		std::printf("[%s] [ERROR] failed to load solver config\n", kModule);
		return 3;
	}

	auto inference = std::make_unique<app::auto_aim::OpenVINOInference>(
	    detector_config.model_path, detector_config.device,
	    detector_config.inference_score_threshold);

	if(!inference->is_ready())
	{
		std::printf("[%s] [ERROR] OpenVINO inference is not ready\n", kModule);
		return 4;
	}

	app::auto_aim::Detector detector(detector_config, std::move(inference));
	app::auto_aim::Solver solver(solver_config);

	app::auto_aim::TrackerConfig tracker_config = app::auto_aim::make_default_tracker_config();
	// 演示：宽松 position gate 便于真实视频自适应；yaw score 保持文档化。
	tracker_config.association.max_position_error_m = 1.0;

	app::auto_aim::Tracker tracker(tracker_config);
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));

	if(!auto_aim.is_ready())
	{
		std::printf("[%s] [ERROR] AutoAim dependencies are not ready\n", kModule);
		return 5;
	}

	std::ofstream csv(csv_path);
	if(!csv.is_open())
	{
		std::printf("[%s] [ERROR] failed to open csv: %s\n", kModule, csv_path.c_str());
		return 6;
	}
	write_csv_header(csv);

	ReplayStats stats;

	cv::Mat frame;
	std::size_t frame_index = 0;

	std::optional<int> prev_matched_armor_id;
	std::optional<Eigen::Vector3d> prev_center;
	app::auto_aim::TrackerState prev_state = app::auto_aim::TrackerState::Lost;

	while(video.read(frame))
	{
		if(frame.empty())
		{
			break;
		}

		app::auto_aim::FrameContext context;
		context.image = frame;

		if(has_sync_quaternions)
		{
			if(frame_index >= quaternions.size())
			{
				break;
			}

			context.timestamp_s = std::get<0>(quaternions[frame_index]);
			context.q_imu_body_to_world = std::get<1>(quaternions[frame_index]);
		}
		else
		{
			context.timestamp_s = static_cast<double>(frame_index) / fps;
		}

		const auto result = auto_aim.process(context);

		++stats.frames;

		if(!result.observations.empty())
		{
			++stats.frames_with_observations;
		}

		if(!result.tracked_target.has_value())
		{
			// Tracker Lost：记 lost transition（仅当之前非 Lost）。
			if(prev_state != app::auto_aim::TrackerState::Lost)
			{
				++stats.lost_transitions;
			}
			prev_state = app::auto_aim::TrackerState::Lost;
			prev_matched_armor_id.reset();
			prev_center.reset();
			++frame_index;
			continue;
		}

		const app::auto_aim::TrackedTarget& tracked = *result.tracked_target;

		write_csv_row(csv, tracked);

		// 状态与初始化/重捕获计数。
		if(prev_state == app::auto_aim::TrackerState::Lost
		   && tracked.state != app::auto_aim::TrackerState::Lost)
		{
			++stats.tracker_initializations;
		}

		if(tracked.state == app::auto_aim::TrackerState::TempLost)
		{
			++stats.temp_lost_frames;
		}

		if(prev_state == app::auto_aim::TrackerState::TempLost
		   && tracked.state == app::auto_aim::TrackerState::Tracking)
		{
			++stats.reacquisitions;
		}

		// association/correction 计数。
		if(tracked.has_measurement)
		{
			++stats.successful_associations;
			++stats.successful_corrections;
		}
		else if(tracked.state == app::auto_aim::TrackerState::Tracking
		        || tracked.state == app::auto_aim::TrackerState::Detecting)
		{
			// 有 target 但本帧未 association 成功：可能是 reject。
			++stats.rejected_associations;
		}

		// board-switch continuity。
		if(tracked.matched_armor_id.has_value())
		{
			if(prev_matched_armor_id.has_value()
			   && *tracked.matched_armor_id != *prev_matched_armor_id)
			{
				++stats.armor_switch_count;

				if(prev_center.has_value() && is_finite_vector3(tracked.center_in_world))
				{
					const double jump = (tracked.center_in_world - *prev_center).norm();
					stats.center_jumps_m.push_back(jump);
				}
			}

			prev_matched_armor_id = tracked.matched_armor_id;
		}

		prev_center = tracked.center_in_world;
		prev_state = tracked.state;

		++frame_index;
	}

	video.release();
	csv.close();

	// ---- quantitative summary ----
	std::printf("\n================= Tracker Replay Summary =================\n");
	std::printf("dataset:                 %s\n", video_path.c_str());
	std::printf("frames:                  %zu\n", stats.frames);
	std::printf("frames_with_observations:%zu\n", stats.frames_with_observations);
	std::printf("successful_associations: %zu\n", stats.successful_associations);
	std::printf("rejected_associations:   %zu\n", stats.rejected_associations);
	std::printf("successful_corrections:  %zu\n", stats.successful_corrections);
	std::printf("correction_failures:     %zu\n", stats.correction_failures);
	std::printf("tracker_initializations: %zu\n", stats.tracker_initializations);
	std::printf("lost_transitions:        %zu\n", stats.lost_transitions);
	std::printf("temp_lost_frames:        %zu\n", stats.temp_lost_frames);
	std::printf("reacquisitions:          %zu\n", stats.reacquisitions);
	std::printf("armor_switch_count:      %zu\n", stats.armor_switch_count);

	if(!stats.center_jumps_m.empty())
	{
		const double mean =
		    std::accumulate(stats.center_jumps_m.begin(), stats.center_jumps_m.end(), 0.0)
		    / static_cast<double>(stats.center_jumps_m.size());
		const double max = *std::max_element(stats.center_jumps_m.begin(),
		                                     stats.center_jumps_m.end());
		const double p95 = percentile(stats.center_jumps_m, 0.95);

		std::printf("center_jump_on_switch:  count=%zu mean=%.3f m max=%.3f m p95=%.3f m\n",
		            stats.center_jumps_m.size(), mean, max, p95);
	}
	else
	{
		std::printf("center_jump_on_switch:  no armor switches observed\n");
	}

	std::printf("csv:                     %s\n", csv_path.c_str());
	std::printf("===========================================================\n");

	// 弱验收：至少处理若干帧并成功初始化 Tracker。
	if(stats.frames == 0)
	{
		std::printf("[%s] [ERROR] no frames processed\n", kModule);
		return 7;
	}

	std::printf("[%s] replay finished\n", kModule);
	return 0;
}