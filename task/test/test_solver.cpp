#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "app/auto_aim/solver.hpp"
#include "tools/maths_tools.hpp"

namespace aa = app::auto_aim;
namespace mt = tools::maths_tools;

namespace
{
	constexpr double kPi = 3.14159265358979323846;

	struct TestRunner
	{
		int passed = 0;
		int failed = 0;

		void expect(bool condition, const std::string& message)
		{
			if(condition)
			{
				++passed;
				std::cout << "[PASS] " << message << '\n';
			}
			else
			{
				++failed;
				std::cerr << "[FAIL] " << message << '\n';
			}
		}

		int result() const
		{
			std::cout << "\n========== Test Summary ==========\n";
			std::cout << "Passed: " << passed << '\n';
			std::cout << "Failed: " << failed << '\n';
			std::cout << "==================================\n";
			return failed == 0 ? 0 : 1;
		}
	};

	bool near(double lhs, double rhs, double eps = 1e-6)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	bool vec_near(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs, double eps = 1e-6)
	{
		return (lhs - rhs).norm() <= eps;
	}

	bool mat_near(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs, double eps = 1e-6)
	{
		if(lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
		{
			return false;
		}
		return (lhs - rhs).norm() <= eps;
	}

	Eigen::MatrixXd numerical_jacobian(
	    const std::function<Eigen::Vector3d(const Eigen::Vector3d&)>& function,
	    const Eigen::Vector3d& x, double step = 1e-6)
	{
		Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 3);

		for(int col = 0; col < 3; ++col)
		{
			Eigen::Vector3d dx = Eigen::Vector3d::Zero();
			dx[col] = step;

			const Eigen::Vector3d y_plus = function(x + dx);
			const Eigen::Vector3d y_minus = function(x - dx);

			jacobian.col(col) = (y_plus - y_minus) / (2.0 * step);
		}

		return jacobian;
	}

	cv::Mat make_camera_matrix()
	{
		return (cv::Mat_<double>(3, 3) << 800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0);
	}

	cv::Mat make_distort_coeffs()
	{
		return cv::Mat::zeros(1, 5, CV_64F);
	}

	std::vector<cv::Point3d> make_small_armor_object_points(double width_m = 135e-3,
	                                                        double lightbar_length_m = 56e-3)
	{
		const double half_width = width_m / 2.0;
		const double half_height = lightbar_length_m / 2.0;

		// 与 Solver::object_points() 保持一致：
		// 左上、右上、右下、左下。
		// 装甲板位于 x = 0 平面。
		return {{0.0, half_width, half_height},
		        {0.0, -half_width, half_height},
		        {0.0, -half_width, -half_height},
		        {0.0, half_width, -half_height}};
	}

	Eigen::Matrix3d make_armor_facing_camera_rotation()
	{
		/*
   * object frame -> camera frame
   *
   * 目标：
   * - 装甲板法向，即 armor x 轴，指向相机 z 轴正方向；
   * - armor y 正方向对应图像左侧，即 camera x 负方向；
   * - armor z 正方向对应图像上方，即 camera y 负方向。
   *
   * OpenCV 相机系：x 向右，y 向下，z 向前。
   */
		Eigen::Matrix3d r;
		r.col(0) = Eigen::Vector3d(0.0, 0.0, 1.0);  // armor x -> camera z
		r.col(1) = Eigen::Vector3d(-1.0, 0.0, 0.0); // armor y -> camera -x
		r.col(2) = Eigen::Vector3d(0.0, -1.0, 0.0); // armor z -> camera -y
		return r;
	}

	std::vector<cv::Point2f> project_small_armor_points(const cv::Mat& camera_matrix,
	                                                    const cv::Mat& distort_coeffs,
	                                                    const cv::Vec3d& tvec)
	{
		const auto object_points = make_small_armor_object_points();

		const Eigen::Matrix3d r_eigen = make_armor_facing_camera_rotation();

		cv::Mat r_cv =
		    (cv::Mat_<double>(3, 3) << r_eigen(0, 0), r_eigen(0, 1), r_eigen(0, 2), r_eigen(1, 0),
		     r_eigen(1, 1), r_eigen(1, 2), r_eigen(2, 0), r_eigen(2, 1), r_eigen(2, 2));

		cv::Mat rvec;
		cv::Rodrigues(r_cv, rvec);

		cv::Mat tvec_mat = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);

		std::vector<cv::Point2d> image_points_double;
		cv::projectPoints(object_points, rvec, tvec_mat, camera_matrix, distort_coeffs,
		                  image_points_double);

		std::vector<cv::Point2f> image_points;
		image_points.reserve(image_points_double.size());

		for(const auto& point: image_points_double)
		{
			image_points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
		}

		return image_points;
	}


	aa::SolverConfig make_valid_solver_config()
	{
		aa::SolverConfig config;
		config.camera_matrix = make_camera_matrix();
		config.distort_coeffs = make_distort_coeffs();

		config.r_gimbal_to_imu_body = Eigen::Matrix3d::Identity();
		config.r_camera_to_gimbal = Eigen::Matrix3d::Identity();
		config.t_camera_to_gimbal = Eigen::Vector3d::Zero();

		config.lightbar_length_m = 56e-3;
		config.small_armor_width_m = 135e-3;
		config.big_armor_width_m = 230e-3;

		return config;
	}

	void test_limit_and_angle_tools(TestRunner& runner)
	{
		runner.expect(near(mt::limit_rad(0.0), 0.0), "limit_rad: 0 -> 0");
		runner.expect(near(mt::limit_rad(2.0 * kPi), 0.0), "limit_rad: 2pi -> 0");
		runner.expect(near(mt::limit_rad(1.5 * kPi), -0.5 * kPi), "limit_rad: 3pi/2 -> -pi/2");
		runner.expect(near(mt::limit_rad(-1.5 * kPi), 0.5 * kPi), "limit_rad: -3pi/2 -> pi/2");

		// 注意：当前实现中 -pi 会被映射成 +pi。
		// 如果你想使用 [-pi, pi) 语义，这里和实现都需要改。
		runner.expect(near(mt::limit_rad(-kPi), kPi), "limit_rad: current behavior maps -pi to +pi");

		runner.expect(near(mt::limit_min_max(5.0, 0.0, 10.0), 5.0), "limit_min_max: inside range");
		runner.expect(near(mt::limit_min_max(-1.0, 0.0, 10.0), 0.0), "limit_min_max: lower clamp");
		runner.expect(near(mt::limit_min_max(12.0, 0.0, 10.0), 10.0), "limit_min_max: upper clamp");

		runner.expect(mt::square(3) == 9, "square<int>");
		runner.expect(near(mt::square(0.5), 0.25), "square<double>");

		runner.expect(
		    near(mt::get_abs_angle(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.0, 1.0)), kPi / 2.0),
		    "get_abs_angle: perpendicular vectors");

		runner.expect(
		    near(mt::get_abs_angle(Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 0.0)), 0.0),
		    "get_abs_angle: zero vector returns 0");
	}

	void test_coordinate_conversion(TestRunner& runner)
	{
		const Eigen::Vector3d xyz(2.0, 1.0, 0.5);

		const Eigen::Vector3d ypd = mt::xyz2ypd(xyz);
		const Eigen::Vector3d xyz_back = mt::ypd2xyz(ypd);

		runner.expect(vec_near(xyz, xyz_back, 1e-8), "xyz2ypd + ypd2xyz closed loop");

		const Eigen::Vector3d ypd_given(0.3, -0.2, 3.0);
		const Eigen::Vector3d xyz_from_ypd = mt::ypd2xyz(ypd_given);
		const Eigen::Vector3d ypd_back = mt::xyz2ypd(xyz_from_ypd);

		runner.expect(vec_near(ypd_given, ypd_back, 1e-8), "ypd2xyz + xyz2ypd closed loop");

		const Eigen::Vector3d ypr(0.3, -0.2, 0.1);
		const Eigen::Matrix3d r = mt::rotation_matrix(ypr);

		runner.expect(mat_near(r.transpose() * r, Eigen::Matrix3d::Identity(), 1e-8),
		              "rotation_matrix: orthonormal");

		runner.expect(near(r.determinant(), 1.0, 1e-8), "rotation_matrix: determinant is 1");
	}

	void test_jacobians(TestRunner& runner)
	{
		const Eigen::Vector3d xyz(1.7, 0.8, 0.6);

		const Eigen::MatrixXd analytic_xyz2ypd = mt::xyz2ypd_jacobian(xyz);
		const Eigen::MatrixXd numeric_xyz2ypd = numerical_jacobian(
		    [](const Eigen::Vector3d& value) {
			    return mt::xyz2ypd(value);
		    },
		    xyz);

		runner.expect(mat_near(analytic_xyz2ypd, numeric_xyz2ypd, 1e-5),
		              "xyz2ypd_jacobian matches numerical differentiation");

		const Eigen::Vector3d ypd(0.35, -0.25, 2.4);

		const Eigen::MatrixXd analytic_ypd2xyz = mt::ypd2xyz_jacobian(ypd);
		const Eigen::MatrixXd numeric_ypd2xyz = numerical_jacobian(
		    [](const Eigen::Vector3d& value) {
			    return mt::ypd2xyz(value);
		    },
		    ypd);

		runner.expect(mat_near(analytic_ypd2xyz, numeric_ypd2xyz, 1e-5),
		              "ypd2xyz_jacobian matches numerical differentiation");
	}

	void test_solver_config_validation(TestRunner& runner)
	{
		aa::SolverConfig invalid_config;
		aa::Solver invalid_solver(invalid_config);

		runner.expect(!invalid_solver.is_valid(), "Solver rejects empty camera matrix");

		aa::SolverConfig valid_config = make_valid_solver_config();
		aa::Solver solver(valid_config);

		runner.expect(solver.is_valid(), "Solver accepts valid config");

		aa::Armor empty_armor;
		empty_armor.type = aa::ArmorType::Small;

		runner.expect(!solver.solve(empty_armor), "Solver rejects armor with less than 4 points");

		aa::Armor unknown_type_armor;
		unknown_type_armor.type = aa::ArmorType::Unknown;
		unknown_type_armor.points = {
		    {600.0F, 350.0F}, {680.0F, 350.0F}, {680.0F, 390.0F}, {600.0F, 390.0F}};

		runner.expect(!solver.solve(unknown_type_armor), "Solver rejects unknown armor type");
	}

	void test_solver_pnp_closed_loop(TestRunner& runner)
	{
		try
		{
			aa::SolverConfig config = make_valid_solver_config();

			config.t_camera_to_gimbal = Eigen::Vector3d(0.01, -0.02, 0.03);

			aa::Solver solver(config);
			runner.expect(solver.is_valid(), "Solver valid before PnP closed-loop test");

			const cv::Vec3d expected_tvec_camera(0.12, -0.04, 2.0);

			aa::Armor armor;
			armor.type = aa::ArmorType::Small;
			armor.points = project_small_armor_points(config.camera_matrix, config.distort_coeffs,
			                                          expected_tvec_camera);

			const bool solved = solver.solve(armor);
			runner.expect(solved, "Solver solves synthetic projected armor");

			if(!solved)
			{
				return;
			}

			const Eigen::Vector3d expected_xyz_camera(
			    expected_tvec_camera[0], expected_tvec_camera[1], expected_tvec_camera[2]);

			const Eigen::Vector3d expected_xyz_gimbal =
			    config.r_camera_to_gimbal * expected_xyz_camera + config.t_camera_to_gimbal;

			runner.expect(vec_near(armor.xyz_in_gimbal, expected_xyz_gimbal, 2e-3),
			              "Solver xyz_in_gimbal matches synthetic tvec plus extrinsic translation");

			runner.expect(
			    vec_near(armor.xyz_in_world, expected_xyz_gimbal, 2e-3),
			    "Solver xyz_in_world equals gimbal coordinates when r_gimbal_to_world is identity");

			runner.expect(vec_near(armor.ypd_in_world, mt::xyz2ypd(expected_xyz_gimbal), 2e-3),
			              "Solver ypd_in_world matches xyz2ypd(xyz_in_world)");
		}
		catch(const cv::Exception& error)
		{
			runner.expect(false,
			              std::string("OpenCV exception in PnP closed-loop test: ") + error.what());
		}
	}


	void test_solver_gimbal_to_world_rotation(TestRunner& runner)
	{
		aa::SolverConfig config = make_valid_solver_config();
		aa::Solver solver(config);

		const Eigen::Quaterniond q_imu_body_to_world(
		    Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitZ()));

		solver.set_r_gimbal_to_world(q_imu_body_to_world);

		const Eigen::Matrix3d expected =
		    Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

		runner.expect(mat_near(solver.r_gimbal_to_world(), expected, 1e-8),
		              "set_r_gimbal_to_world with identity r_gimbal_to_imu_body");
	}

	/**
	 * @brief 回归测试：锁定旧 sp_vision_25 Solver 的 r_gimbal_to_world 行为。
	 *
	 * 当前迁移代码保持旧行为：
	 *   r_gimbal_to_world =
	 *       r_gimbal_to_imu_body^T
	 *       * r_imu_body_to_world
	 *       * r_gimbal_to_imu_body
	 *
	 * 该公式的物理语义是否合理，属于独立的坐标系审计阶段；
	 * 本测试只用于防止迁移过程中"顺手修正"而改变旧行为。
	 */
	void test_solver_gimbal_to_world_regression(TestRunner& runner)
	{
		aa::SolverConfig config = make_valid_solver_config();

		// 非单位阵的 gimbal -> imu_body 旋转。
		config.r_gimbal_to_imu_body =
		    Eigen::AngleAxisd(kPi / 3.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

		aa::Solver solver(config);

		// 非单位四元数：绕 X 旋转 40°。
		const Eigen::Quaterniond q_imu_body_to_world(
		    Eigen::AngleAxisd(40.0 * kPi / 180.0, Eigen::Vector3d::UnitX()));

		solver.set_r_gimbal_to_world(q_imu_body_to_world);

		const Eigen::Matrix3d r_imu_body_to_world =
		    q_imu_body_to_world.normalized().toRotationMatrix();

		const Eigen::Matrix3d expected = config.r_gimbal_to_imu_body.transpose()
		    * r_imu_body_to_world * config.r_gimbal_to_imu_body;

		runner.expect(
		    mat_near(solver.r_gimbal_to_world(), expected, 1e-8),
		    "non-identity r_gimbal_to_imu_body: current formula R^T * R_imu2world * R");
	}

} // namespace

int main()
{
	TestRunner runner;

	test_limit_and_angle_tools(runner);
	test_coordinate_conversion(runner);
	test_jacobians(runner);
	test_solver_config_validation(runner);
	test_solver_pnp_closed_loop(runner);
	test_solver_gimbal_to_world_rotation(runner);
	test_solver_gimbal_to_world_regression(runner);

	return runner.result();
}
