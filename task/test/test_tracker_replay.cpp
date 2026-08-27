/**
 * @file test_tracker_replay.cpp
 * @brief Tracker v1 真实/录制数据 replay + 结构化诊断 + quantitative validation。
 *
 * manual integration/replay executable，不注册进默认 CTest。
 *
 * 输出：
 * - data/logs/tracker_replay.csv   （每帧一行，含 Lost 帧）
 * - stdout quantitative summary + coverage 报告
 *
 * 区分：
 * - execution success（进程退出码仅表示执行是否成功）
 * - validation coverage（由 coverage 报告单独呈现，缺失场景打印 NOT COVERED）
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"
#include "app/auto_aim/tracker_config.hpp"

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
#include "test_logging.hpp"

namespace
{

	constexpr const char* kModule = "TRACKER_REPLAY";

	// 每个正确 correction 的 center jump 分布。
	struct Distribution
	{
		std::vector<double> values;
	};

	struct ReplayStats
	{
		std::size_t frames = 0;
		std::size_t frames_with_observations = 0;

		std::size_t initialized = 0;
		std::size_t corrected = 0;
		std::size_t no_association = 0;
		std::size_t correction_failed = 0;

		std::size_t lost_transitions = 0;
		std::size_t temp_lost_to_lost = 0;
		std::size_t temp_lost_frames = 0;
		std::size_t reacquisitions = 0;

		std::size_t armor_switch_count = 0;

		Distribution correction_jumps_all;
		Distribution correction_jumps_non_switch;
		Distribution correction_jumps_switch;
	};

	bool is_finite_vector3(const Eigen::Vector3d& v)
	{
		return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
	}

	bool is_finite_quaternion(double w, double x, double y, double z, double timestamp)
	{
		return std::isfinite(w) && std::isfinite(x) && std::isfinite(y) && std::isfinite(z)
		    && std::isfinite(timestamp);
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
			// P2-2：normalize 前显式拒绝任何非 finite 分量。
			if(!is_finite_quaternion(w, x, y, z, timestamp))
			{
				std::printf("[%s] [WARN] non-finite quaternion/timestamp; falling back to identity\n",
				            kModule);
				return false;
			}

			const double norm2 = w * w + x * x + y * y + z * z;

			if(!std::isfinite(norm2) || norm2 <= 1e-12)
			{
				std::printf("[%s] [WARN] zero-norm quaternion; falling back to identity\n", kModule);
				return false;
			}

			Eigen::Quaterniond q(w, x, y, z);
			q.normalize();
			out.emplace_back(timestamp, q);
		}

		return !out.empty();
	}

	const char* state_name(app::auto_aim::TrackerState state)
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

	const char* outcome_name(app::auto_aim::TrackUpdateOutcome outcome)
	{
		switch(outcome)
		{
		case app::auto_aim::TrackUpdateOutcome::NotTracked:
			return "NotTracked";
		case app::auto_aim::TrackUpdateOutcome::Initialized:
			return "Initialized";
		case app::auto_aim::TrackUpdateOutcome::Corrected:
			return "Corrected";
		case app::auto_aim::TrackUpdateOutcome::NoAssociation:
			return "NoAssociation";
		case app::auto_aim::TrackUpdateOutcome::CorrectionFailed:
			return "CorrectionFailed";
		default:
			return "Unknown";
		}
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

	void print_distribution(const char* label, const Distribution& distribution)
	{
		const std::vector<double>& values = distribution.values;

		if(values.empty())
		{
			std::printf("  %-28s count=0 mean=NaN m p95=NaN m max=NaN m\n", label);
			return;
		}

		const double mean = std::accumulate(values.begin(), values.end(), 0.0)
		    / static_cast<double>(values.size());
		const double p95 = percentile(values, 0.95);
		const double max = *std::max_element(values.begin(), values.end());

		std::printf("  %-28s count=%zu mean=%.6f m p95=%.6f m max=%.6f m\n", label, values.size(),
		            mean, p95, max);
	}

	// ----------------- CSV（列对齐保证 post-hoc 可重建）-----------------

	std::vector<std::string> base_fields(double timestamp_s,
	                                     app::auto_aim::TrackUpdateOutcome outcome,
	                                     const app::auto_aim::TrackedTarget* t,
	                                     const std::vector<app::auto_aim::ArmorObservation>* obs)
	{
		std::vector<std::string> f;

		f.push_back(std::to_string(timestamp_s));

		if(t != nullptr)
		{
			f.push_back(state_name(t->state));
			f.push_back(outcome_name(outcome));
			f.push_back(t->has_measurement ? "1" : "0");
		}
		else
		{
			f.push_back("Lost");
			f.push_back(outcome_name(outcome));
			f.push_back("0");
		}

		// obs_count = 真实 observations.size()。
		const std::size_t obs_count = (obs != nullptr) ? obs->size() : 0;
		f.push_back(std::to_string(obs_count));

		// 固定两个 observation 槽（保持列数固定）。
		for(std::size_t i = 0; i < 2; ++i)
		{
			if(obs != nullptr && i < obs->size())
			{
				const auto& o = (*obs)[i];
				f.push_back(std::to_string(o.position_in_world.x()));
				f.push_back(std::to_string(o.position_in_world.y()));
				f.push_back(std::to_string(o.position_in_world.z()));
				f.push_back(std::to_string(o.armor_yaw_in_world));
			}
			else
			{
				f.push_back("");
				f.push_back("");
				f.push_back("");
				f.push_back("");
			}
		}

		// matched observation index / armor id。
		if(t != nullptr)
		{
			f.push_back(t->matched_observation_index.has_value()
			                ? std::to_string(*t->matched_observation_index)
			                : "");
			f.push_back(t->matched_armor_id.has_value() ? std::to_string(*t->matched_armor_id) : "");
		}
		else
		{
			f.push_back("");
			f.push_back("");
		}

		// matched observation xyz/yaw（固定 4 列；有 matched index 时回填真实观测）。
		const bool has_matched_obs = t != nullptr && t->matched_observation_index.has_value()
		    && obs != nullptr && *t->matched_observation_index < obs->size();
		for(int k = 0; k < 4; ++k)
		{
			if(has_matched_obs)
			{
				const auto& o = (*obs)[*t->matched_observation_index];

				switch(k)
				{
				case 0:
					f.push_back(std::to_string(o.position_in_world.x()));
					break;
				case 1:
					f.push_back(std::to_string(o.position_in_world.y()));
					break;
				case 2:
					f.push_back(std::to_string(o.position_in_world.z()));
					break;
				default:
					f.push_back(std::to_string(o.armor_yaw_in_world));
					break;
				}
			}
			else
			{
				f.push_back("");
			}
		}

		if(t != nullptr)
		{
			f.push_back(std::to_string(t->prior_predicted_center.x()));
			f.push_back(std::to_string(t->prior_predicted_center.y()));
			f.push_back(std::to_string(t->prior_predicted_center.z()));
			f.push_back(std::to_string(t->center_in_world.x()));
			f.push_back(std::to_string(t->center_in_world.y()));
			f.push_back(std::to_string(t->center_in_world.z()));
			f.push_back(std::to_string(t->velocity_in_world.x()));
			f.push_back(std::to_string(t->velocity_in_world.y()));
			f.push_back(std::to_string(t->velocity_in_world.z()));
			f.push_back(std::to_string(t->yaw));
			f.push_back(std::to_string(t->yaw_rate));
			f.push_back(std::to_string(t->radius));
			f.push_back(std::to_string(t->delta_radius));
			f.push_back(std::to_string(t->delta_z));
			f.push_back(std::to_string(t->predicted_armors.size()));

			// 固定 4 组 armor hypothesis 本体（id + x + y + z + yaw）。
			for(int armor_id = 0; armor_id < 4; ++armor_id)
			{
				const app::auto_aim::ArmorHypothesis* hypothesis = nullptr;

				for(const auto& h: t->predicted_armors)
				{
					if(h.armor_id == armor_id)
					{
						hypothesis = &h;
						break;
					}
				}

				if(hypothesis != nullptr)
				{
					f.push_back(std::to_string(hypothesis->armor_id));
					f.push_back(std::to_string(hypothesis->position_in_world.x()));
					f.push_back(std::to_string(hypothesis->position_in_world.y()));
					f.push_back(std::to_string(hypothesis->position_in_world.z()));
					f.push_back(std::to_string(hypothesis->yaw_in_world));
				}
				else
				{
					f.push_back("");
					f.push_back("");
					f.push_back("");
					f.push_back("");
					f.push_back("");
				}
			}

			if(t->innovation.has_value() && t->innovation->size() == 4)
			{
				f.push_back(std::to_string((*t->innovation)(0)));
				f.push_back(std::to_string((*t->innovation)(1)));
				f.push_back(std::to_string((*t->innovation)(2)));
				f.push_back(std::to_string((*t->innovation)(3)));
			}
			else
			{
				f.push_back("");
				f.push_back("");
				f.push_back("");
				f.push_back("");
			}

			f.push_back(t->nis.has_value() ? std::to_string(*t->nis) : "");
		}
		else
		{
			// prior(3) + center(3) + vel(3) + yaw/yaw_rate/radius/dr/dz(5) + pred_armors(1)
			// + hypothesis(20) + innovation(4) + nis(1) = 40 个字段。
			for(int i = 0; i < 40; ++i)
			{
				f.push_back("");
			}
		}

		return f;
	}

	void write_csv_header(std::ofstream& out)
	{
		out << "timestamp_s,state,outcome,has_measurement,obs_count,"
		    << "obs0_x,obs0_y,obs0_z,obs0_yaw,obs1_x,obs1_y,obs1_z,obs1_yaw,"
		    << "matched_observation_index,matched_armor_id,"
		    << "matched_obs_x,matched_obs_y,matched_obs_z,matched_obs_yaw,"
		    << "prior_cx,prior_cy,prior_cz,"
		    << "center_x,center_y,center_z,"
		    << "vel_x,vel_y,vel_z,"
		    << "yaw,yaw_rate,radius,delta_radius,delta_z,pred_armors,"
		    << "pred0_id,pred0_x,pred0_y,pred0_z,pred0_yaw,"
		    << "pred1_id,pred1_x,pred1_y,pred1_z,pred1_yaw,"
		    << "pred2_id,pred2_x,pred2_y,pred2_z,pred2_yaw,"
		    << "pred3_id,pred3_x,pred3_y,pred3_z,pred3_yaw,"
		    << "innovation_x,innovation_y,innovation_z,innovation_yaw,nis\n";
	}

	void write_row(std::ofstream& out, double timestamp_s,
	               app::auto_aim::TrackUpdateOutcome outcome,
	               const app::auto_aim::TrackedTarget* t,
	               const std::vector<app::auto_aim::ArmorObservation>* observations)
	{
		const auto fields = base_fields(timestamp_s, outcome, t, observations);

		for(std::size_t i = 0; i < fields.size(); ++i)
		{
			if(i != 0)
			{
				out << ',';
			}

			out << fields[i];
		}

		out << '\n';
	}

} // namespace

int main()
{
	test_logging::init("test_tracker_replay");
	const std::string project_root = std::string(PROJECT_SOURCE_DIR) + "/";

	const std::string video_path = project_root + "data/demo/demo.avi";
	const std::string quaternion_path = project_root + "data/demo/demo.txt";
	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";
	const std::string tracker_config_path = project_root + "config/app/auto_aim/tracker.toml";
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
		std::printf("[%s] [WARN] no valid synchronized quaternion data; using identity\n", kModule);
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

	// P1-2：生产 baseline 统一加载同一 tracker.toml，不做 replay-only 静默 gate 扩大。
	app::auto_aim::TrackerConfig tracker_config;
	if(!app::auto_aim::load_tracker_config(tracker_config_path, tracker_config))
	{
		std::printf("[%s] [ERROR] failed to load tracker config\n", kModule);
		return 4;
	}

	auto inference = std::make_unique<app::auto_aim::OpenVINOInference>(
	    detector_config.model_path, detector_config.device,
	    detector_config.inference_score_threshold);

	if(!inference->is_ready())
	{
		std::printf("[%s] [ERROR] OpenVINO inference is not ready\n", kModule);
		return 5;
	}

	app::auto_aim::Detector detector(detector_config, std::move(inference));
	app::auto_aim::Solver solver(solver_config);
	app::auto_aim::Tracker tracker(tracker_config);
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker),
	                                app::auto_aim::Aimer(app::auto_aim::make_default_aimer_config()),
	                                app::auto_aim::Shooter(app::auto_aim::make_default_shooter_config()));

	if(!auto_aim.is_ready())
	{
		std::printf("[%s] [ERROR] AutoAim dependencies are not ready\n", kModule);
		return 6;
	}

	std::ofstream csv(csv_path);
	if(!csv.is_open())
	{
		std::printf("[%s] [ERROR] failed to open csv: %s\n", kModule, csv_path.c_str());
		return 7;
	}
	write_csv_header(csv);

	ReplayStats stats;

	cv::Mat frame;
	std::size_t frame_index = 0;

	std::optional<int> prev_matched_armor_id;
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

		// outcome 计数按 result.outcome（含 Lost 帧的真实 outcome）。
		switch(result.outcome)
		{
		case app::auto_aim::TrackUpdateOutcome::Initialized:
			++stats.initialized;
			break;
		case app::auto_aim::TrackUpdateOutcome::Corrected:
			++stats.corrected;
			break;
		case app::auto_aim::TrackUpdateOutcome::NoAssociation:
			++stats.no_association;
			break;
		case app::auto_aim::TrackUpdateOutcome::CorrectionFailed:
			++stats.correction_failed;
			break;
		case app::auto_aim::TrackUpdateOutcome::NotTracked:
			break;
		}

		const bool has_tracked = result.tracked_target.has_value();

		// 写行：Lost 行也要带上 observations。
		if(has_tracked)
		{
			write_row(csv, context.timestamp_s, result.outcome, &*result.tracked_target,
			          &result.observations);
		}
		else
		{
			write_row(csv, context.timestamp_s, result.outcome, nullptr, &result.observations);
		}

		// 状态转移计数。
		if(has_tracked)
		{
			const app::auto_aim::TrackedTarget& tracked = *result.tracked_target;

			if(tracked.state == app::auto_aim::TrackerState::TempLost)
			{
				++stats.temp_lost_frames;
			}

			if(prev_state == app::auto_aim::TrackerState::TempLost
			   && tracked.state == app::auto_aim::TrackerState::Tracking)
			{
				++stats.reacquisitions;
			}

			// board-switch continuity 三分布（仅 Corrected）。
			if(tracked.matched_armor_id.has_value()
			   && result.outcome == app::auto_aim::TrackUpdateOutcome::Corrected
			   && is_finite_vector3(tracked.center_in_world)
			   && is_finite_vector3(tracked.prior_predicted_center))
			{
				const double jump =
				    (tracked.center_in_world - tracked.prior_predicted_center).norm();

				stats.correction_jumps_all.values.push_back(jump);

				const bool is_switch = prev_matched_armor_id.has_value()
				    && *tracked.matched_armor_id != *prev_matched_armor_id;

				if(is_switch)
				{
					++stats.armor_switch_count;
					stats.correction_jumps_switch.values.push_back(jump);
				}
				else
				{
					stats.correction_jumps_non_switch.values.push_back(jump);
				}

				prev_matched_armor_id = tracked.matched_armor_id;
			}

			prev_state = tracked.state;
		}
		else
		{
			// Lost：观测丢失，重置 board 连续性记忆，并统计转移。
			if(prev_state == app::auto_aim::TrackerState::TempLost)
			{
				++stats.temp_lost_to_lost;
			}

			if(prev_state != app::auto_aim::TrackerState::Lost)
			{
				++stats.lost_transitions;
			}

			prev_state = app::auto_aim::TrackerState::Lost;
			prev_matched_armor_id.reset();
		}

		++frame_index;
	}

	video.release();
	csv.close();

	// ---- quantitative summary ----
	std::printf("\n================= Tracker Replay Summary =================\n");
	std::printf("dataset:                 %s\n", video_path.c_str());
	std::printf("frames:                  %zu\n", stats.frames);
	std::printf("frames_with_observations:%zu\n", stats.frames_with_observations);
	std::printf("initialized:             %zu\n", stats.initialized);
	std::printf("corrected:               %zu\n", stats.corrected);
	std::printf("no_association:          %zu\n", stats.no_association);
	std::printf("correction_failed:       %zu\n", stats.correction_failed);
	std::printf("lost_transitions:        %zu\n", stats.lost_transitions);
	std::printf("temp_lost_to_lost:       %zu\n", stats.temp_lost_to_lost);
	std::printf("temp_lost_frames:        %zu\n", stats.temp_lost_frames);
	std::printf("reacquisitions:          %zu\n", stats.reacquisitions);
	std::printf("armor_switch_count:      %zu\n", stats.armor_switch_count);

	std::printf("\ncorrection center jump distributions:\n");
	print_distribution("all corrections", stats.correction_jumps_all);
	print_distribution("non-switch corrections", stats.correction_jumps_non_switch);
	print_distribution("switch corrections", stats.correction_jumps_switch);

	std::printf("\ncsv:                     %s\n", csv_path.c_str());
	std::printf("===========================================================\n");

	// ---- validation coverage（P1-8，与执行成功分离）----
	std::printf("\n================= Validation Coverage =================\n");
	auto covered = [](const char* label, bool condition) {
		std::printf("  %-30s %s\n", label, condition ? "COVERED" : "NOT COVERED BY AVAILABLE DATA");
	};

	covered("tracker initialized", stats.initialized > 0);
	covered("successful corrections", stats.corrected > 0);
	covered("armor switches observed", stats.armor_switch_count > 0);
	covered("TempLost observed", stats.temp_lost_frames > 0);
	covered("reacquisition observed", stats.reacquisitions > 0);
	covered("long-loss observed", stats.temp_lost_to_lost > 0);
	std::printf("===========================================================\n");

	if(stats.frames == 0)
	{
		std::printf("[%s] [ERROR] no frames processed\n", kModule);
		return 8;
	}

	std::printf("[%s] replay finished (execution succeeded)\n", kModule);
	return 0;
}