/**
 * @file test_all_pipeline.cpp
 * @brief demo.avi 真实数据集成测试：Detector + Solver + AutoAim + ExtendedKalmanFilter。
 *
 * 数据流：
 *   data/demo/demo.avi + demo.txt(quaternion)
 *     -> OpenVINOInference -> Detector -> Solver -> AutoAim::process
 *     -> AimResult::observations / target
 *     -> [本测试内部定义的 constant-velocity 位置模型]
 *     -> ExtendedKalmanFilter: predict(dt) + update(position_in_gimbal)
 *
 * 重要边界：
 * - 本测试只演示通用 EKF 在真实测量序列上的 predict/update 集成。
 * - 不实现 Target 模型、association、Tracker 状态机、弹道提前量。
 * - 状态模型（gimbal 系 6D constant-velocity）仅在本文件内定义，
 *   不新增生产文件。
 *
 * 断言为弱断言 + 数值健康检查（真实视频无 ground-truth，不做精度阈值）。
 */

#include "app/auto_aim/auto_aim.hpp"
#include "app/auto_aim/detector/detector_config.hpp"
#include "app/auto_aim/detector/openvino_inference.hpp"
#include "app/auto_aim/solver_config.hpp"
#include "tools/extended_kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "test_logging.hpp"

namespace
{

	constexpr const char* kModule = "TEST_ALL_PIPELINE";

	constexpr double kMaxDtS = 0.5; // 首帧/断帧大 gap 的 dt clamp 上限

	bool is_finite_vector3(const Eigen::Vector3d& v)
	{
		return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
	}

	/**
	 * @brief 尝试加载与视频同步的 quaternion 文件（timestamp w x y z）。
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
			if(!std::isfinite(timestamp) || !std::isfinite(w) || !std::isfinite(x)
			   || !std::isfinite(y) || !std::isfinite(z))
			{
				return false;
			}

			const double norm_squared = w * w + x * x + y * y + z * z;

			if(!std::isfinite(norm_squared) || norm_squared <= 1e-12)
			{
				return false;
			}

			Eigen::Quaterniond q(w, x, y, z);
			q.normalize();
			out.emplace_back(timestamp, q);
			++count;
		}

		return count > 0;
	}

	// 构建 6D constant-velocity 状态转移矩阵：状态 [x,y,z,vx,vy,vz]。
	Eigen::MatrixXd make_transition_matrix(double dt)
	{
		Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);

		F(0, 3) = dt;
		F(1, 4) = dt;
		F(2, 5) = dt;

		return F;
	}

	Eigen::MatrixXd make_process_noise()
	{
		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);

		// 仅为演示用的固定小对角值，不属于调参。
		Q(0, 0) = 1e-4;
		Q(1, 1) = 1e-4;
		Q(2, 2) = 1e-4;
		Q(3, 3) = 1e-2;
		Q(4, 4) = 1e-2;
		Q(5, 5) = 1e-2;

		return Q;
	}

	Eigen::MatrixXd make_measurement_matrix()
	{
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
		H(0, 0) = 1.0;
		H(1, 1) = 1.0;
		H(2, 2) = 1.0;

		return H;
	}

	Eigen::MatrixXd make_measurement_noise()
	{
		Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, 3);

		R(0, 0) = 1e-2;
		R(1, 1) = 1e-2;
		R(2, 2) = 1e-2;

		return R;
	}

	// 在画面上叠加：装甲板轮廓、中心点、测量/滤波位置文本。
	void draw_overlay(cv::Mat& canvas, const app::auto_aim::AimResult& result,
	                  const Eigen::VectorXd& state, std::size_t frame_index, double nis)
	{
		// 检测到的装甲板轮廓（绿色）。
		if(result.has_visible_target && result.target.points.size() == 4)
		{
			std::vector<cv::Point> contour;
			contour.reserve(4);

			for(const auto& p: result.target.points)
			{
				contour.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
			}

			cv::polylines(canvas, contour, true, cv::Scalar(0, 255, 0), 2);

			const cv::Point center(static_cast<int>(result.target.center.x),
			                       static_cast<int>(result.target.center.y));
			cv::circle(canvas, center, 5, cv::Scalar(0, 255, 0), -1);
		}

		// 状态信息（左上角）。
		char line0[256];
		std::snprintf(line0, sizeof(line0), "frame %zu", frame_index);
		cv::putText(canvas, line0, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		            cv::Scalar(255, 255, 255), 1);

		if(result.has_visible_target)
		{
			char line1[256];
			std::snprintf(line1, sizeof(line1), "meas (%.3f, %.3f, %.3f)",
			              result.target.xyz_in_gimbal.x(), result.target.xyz_in_gimbal.y(),
			              result.target.xyz_in_gimbal.z());
			cv::putText(canvas, line1, cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(0, 255, 255), 1);
		}

		char line2[256];
		std::snprintf(line2, sizeof(line2), "filter (%.3f, %.3f, %.3f)", state(0), state(1),
		              state(2));
		cv::putText(canvas, line2, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		            cv::Scalar(255, 0, 255), 1);

		char line3[256];
		std::snprintf(line3, sizeof(line3), "vel (%.3f, %.3f, %.3f)  nis=%.3f", state(3),
		              state(4), state(5), nis);
		cv::putText(canvas, line3, cv::Point(10, 105), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		            cv::Scalar(200, 200, 200), 1);
	}

} // namespace

int main(int argc, char** argv)
{
	test_logging::init("test_all_pipeline");
	const bool visual = [&]() {
		for(int i = 1; i < argc; ++i)
		{
			if(std::string(argv[i]) == "--visual")
			{
				return true;
			}
		}

		return false;
	}();

	const std::string project_root = std::string(PROJECT_SOURCE_DIR) + "/";

	const std::string video_path = project_root + "data/demo/demo.avi";
	const std::string quaternion_path = project_root + "data/demo/demo.txt";
	const std::string detector_config_path = project_root + "config/app/auto_aim/detector.toml";
	const std::string solver_config_path = project_root + "config/app/auto_aim/solver_demo.toml";

	// ---- 1. 视频与 quaternion ----
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
		std::printf("[%s] [WARN] no synchronized quaternion data, using identity fallback\n",
		            kModule);
	}

	// ---- 2. 配置 ----
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

	// ---- 3. 构建 pipeline ----
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
	app::auto_aim::Tracker tracker(app::auto_aim::make_default_tracker_config());
	app::auto_aim::AutoAim auto_aim(std::move(detector), std::move(solver), std::move(tracker));

	if(!auto_aim.is_ready())
	{
		std::printf("[%s] [ERROR] AutoAim dependencies are not ready\n", kModule);
		return 5;
	}

	// ---- 4. 通用 EKF（gimbal 6D constant-velocity，仅测试内使用）----
	tools::ExtendedKalmanFilter ekf(
	    Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));

	const Eigen::MatrixXd H = make_measurement_matrix();
	const Eigen::MatrixXd R = make_measurement_noise();
	const Eigen::MatrixXd Q = make_process_noise();

	bool ekf_initialized = false;
	double prev_timestamp_s = 0.0;

	std::size_t processed_frames = 0;
	std::size_t update_count = 0;
	std::size_t target_frames = 0;

	// ---- 5. 逐帧处理 ----
	cv::Mat frame;
	std::size_t frame_index = 0;

	while(true)
	{
		if(!video.read(frame) || frame.empty())
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

		++processed_frames;

		// 有 target：用 gimbal 位置作为测量做 update；否则只 predict。
		if(result.has_visible_target && is_finite_vector3(result.target.xyz_in_gimbal))
		{
			++target_frames;

			const Eigen::Vector3d position = result.target.xyz_in_gimbal;

			if(!ekf_initialized)
			{
				Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);
				x0.head<3>() = position;

				Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(6, 6);
				p0(0, 0) = 1.0;
				p0(1, 1) = 1.0;
				p0(2, 2) = 1.0;
				p0(3, 3) = 100.0;
				p0(4, 4) = 100.0;
				p0(5, 5) = 100.0;

				ekf.reset(x0, p0);
				ekf_initialized = true;
			}
			else
			{
				double dt = context.timestamp_s - prev_timestamp_s;
				dt = std::clamp(dt, 0.0, kMaxDtS);

				ekf.predict(make_transition_matrix(dt), Q);

				Eigen::VectorXd z(3);
				z << position.x(), position.y(), position.z();

				if(ekf.update(z, H, R))
				{
					++update_count;
				}
			}

			prev_timestamp_s = context.timestamp_s;
		}
		else if(ekf_initialized)
		{
			double dt = context.timestamp_s - prev_timestamp_s;
			dt = std::clamp(dt, 0.0, kMaxDtS);

			ekf.predict(make_transition_matrix(dt), Q);
			prev_timestamp_s = context.timestamp_s;
		}

		// 采样输出。
		if(frame_index % 60 == 0 || frame_index == 0)
		{
			const Eigen::VectorXd& x = ekf.state();
			std::printf(
			    "[%s] frame %zu: target=%d state=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f] "
			    "nis=%.3f\n",
			    kModule, frame_index, result.has_visible_target ? 1 : 0, x(0), x(1), x(2), x(3),
			    x(4),
			    x(5), ekf.last_nis());
		}

		// 可视化显示（仅 --visual）。
		if(visual)
		{
			cv::Mat canvas = frame.clone();
			draw_overlay(canvas, result, ekf.state(), frame_index, ekf.last_nis());

			cv::imshow("all_pipeline", canvas);

			const int wait_ms = std::max(1, static_cast<int>(std::round(1000.0 / fps)));
			const int key = cv::waitKey(wait_ms);

			if(key == 'q' || key == 'Q' || key == 27)
			{
				++frame_index;
				break;
			}
		}

		++frame_index;
	}

	video.release();

	if(visual)
	{
		cv::destroyAllWindows();
	}

	// ---- 6. 弱断言 + 数值健康检查 ----
	std::printf("\n================= All Pipeline Summary =================\n");
	std::printf("processed_frames: %zu\n", processed_frames);
	std::printf("target_frames:    %zu\n", target_frames);
	std::printf("update_count:     %zu\n", update_count);
	std::printf("========================================================\n");

	int failures = 0;

	auto expect = [&failures](bool condition, const char* message) {
		if(condition)
		{
			std::printf("[PASS] %s\n", message);
		}
		else
		{
			++failures;
			std::printf("[FAIL] %s\n", message);
		}
	};

	expect(processed_frames > 0, "processed at least one frame");
	expect(ekf_initialized, "EKF was initialized from at least one target");
	expect(update_count > 0, "at least one measurement update succeeded");

	if(ekf_initialized)
	{
		const Eigen::VectorXd& x = ekf.state();
		const Eigen::MatrixXd& p = ekf.covariance();

		expect(x.allFinite(), "final EKF state is finite");
		expect(p.allFinite(), "final EKF covariance is finite");
		expect((p - p.transpose()).norm() < 1e-6,
		       "final EKF covariance is approximately symmetric");
	}

	if(failures != 0)
	{
		std::printf("=== test_all_pipeline failed (%d) ===\n", failures);
		return 1;
	}

	std::printf("=== test_all_pipeline passed ===\n");
	return 0;
}