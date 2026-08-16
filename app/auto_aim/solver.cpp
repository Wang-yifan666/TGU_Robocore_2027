#include "app/auto_aim/solver.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "tools/maths_tools.hpp"

namespace app::auto_aim
{

	namespace
	{
		bool check_camera_matrix(const cv::Mat& camera_matrix)
		{
			return !camera_matrix.empty() && camera_matrix.rows == 3 && camera_matrix.cols == 3;
		}

		Eigen::Matrix3d cv_rotation_to_eigen(const cv::Mat& rotation)
		{
			Eigen::Matrix3d result = Eigen::Matrix3d::Identity();

			for(int row = 0; row < 3; ++row)
			{
				for(int col = 0; col < 3; ++col)
				{
					result(row, col) = rotation.at<double>(row, col);
				}
			}

			return result;
		}

	} // namespace

	Solver::Solver(const SolverConfig& config):
	config_(config), camera_matrix_(config.camera_matrix.clone()),
	distort_coeffs_(config.distort_coeffs.clone())
	{
		valid_ = check_camera_matrix(camera_matrix_) && config_.lightbar_length_m > 0.0
		    && config_.small_armor_width_m > 0.0 && config_.big_armor_width_m > 0.0;

		if(!valid_)
		{
			return;
		}

		camera_matrix_.convertTo(camera_matrix_, CV_64F);

		if(!distort_coeffs_.empty())
		{
			distort_coeffs_.convertTo(distort_coeffs_, CV_64F);
		}
	}

	bool Solver::is_valid() const noexcept
	{
		return valid_;
	}

	const Eigen::Matrix3d& Solver::r_gimbal_to_world() const noexcept
	{
		return r_gimbal_to_world_;
	}

	void Solver::set_r_gimbal_to_world(const Eigen::Quaterniond& q_imu_body_to_world)
	{
		const Eigen::Matrix3d r_imu_body_to_world =
		    q_imu_body_to_world.normalized().toRotationMatrix();

		r_gimbal_to_world_ = config_.r_gimbal_to_imu_body.transpose() * r_imu_body_to_world
		    * config_.r_gimbal_to_imu_body;
	}

	std::vector<cv::Point3d> Solver::object_points(ArmorType type) const
	{
		const double width =
		    type == ArmorType::Big ? config_.big_armor_width_m : config_.small_armor_width_m;

		const double half_width = width / 2.0;
		const double half_height = config_.lightbar_length_m / 2.0;

		/*
         * 与当前 Armor::points 顺序保持一致：
         * 左上、右上、右下、左下。
         *
         * 装甲板位于 x = 0 平面。
         */
		return {{0.0, half_width, half_height},
		        {0.0, -half_width, half_height},
		        {0.0, -half_width, -half_height},
		        {0.0, half_width, -half_height}};
	}

	bool Solver::solve(Armor& armor) const
	{
		if(!valid_)
		{
			return false;
		}

		if(armor.points.size() != 4)
		{
			return false;
		}

		if(armor.type != ArmorType::Small && armor.type != ArmorType::Big)
		{
			return false;
		}

		const auto model_points = object_points(armor.type);

		cv::Vec3d rvec;
		cv::Vec3d tvec;

		bool solved = false;

		try
		{
			solved = cv::solvePnP(model_points, armor.points, camera_matrix_, distort_coeffs_, rvec,
			                      tvec, false, cv::SOLVEPNP_IPPE);
		}
		catch(const cv::Exception&)
		{
			return false;
		}

		if(!solved)
		{
			return false;
		}

		/*
        * OpenCV 相机坐标系中 z 轴指向镜头前方。
        * z <= 0 表示解在相机后方，应视为非法结果。
        *
        * 防御性校验：退化相机矩阵等输入可能导致 solvePnP 返回
        * true 但 tvec 含有 NaN/Inf。这里显式拒绝非有限结果，
        * 保证调用方拿到的 xyz / ypr / ypd 一定有限。
        *
        * 注意：此校验不改变任何坐标变换公式。
        */
		if(!std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) || !std::isfinite(tvec[2])
		   || tvec[2] <= 0.0)
		{
			return false;
		}

		const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};

		armor.xyz_in_gimbal =
		    config_.r_camera_to_gimbal * xyz_in_camera + config_.t_camera_to_gimbal;

		armor.xyz_in_world = r_gimbal_to_world_ * armor.xyz_in_gimbal;

		cv::Mat r_armor_to_camera_cv;
		cv::Rodrigues(rvec, r_armor_to_camera_cv);

		const Eigen::Matrix3d r_armor_to_camera = cv_rotation_to_eigen(r_armor_to_camera_cv);

		const Eigen::Matrix3d r_armor_to_gimbal = config_.r_camera_to_gimbal * r_armor_to_camera;

		const Eigen::Matrix3d r_armor_to_world = r_gimbal_to_world_ * r_armor_to_gimbal;

		/*
        * 当前采用 Z-Y-X 顺序表达 yaw / pitch / roll。
        * 后续接入旧工程 yaw 优化时，应统一检查角度方向和正负号。
        */
		armor.ypr_in_gimbal = tools::maths_tools::rot_to_euler(r_armor_to_gimbal, 2, 1, 0);
		armor.ypr_in_world = tools::maths_tools::rot_to_euler(r_armor_to_world, 2, 1, 0);

		armor.ypd_in_world = tools::maths_tools::xyz2ypd(armor.xyz_in_world);

		armor.yaw_raw = armor.ypr_in_world.x();

		return true;
	}

} // namespace app::auto_aim
