/**
 * @file solver.hpp
 * @brief PnP 求解器，计算装甲板相对于云台/世界的空间位置与姿态。
 *
 *   处理：
 * - 通过 PnP 求解装甲板的旋转向量 rvec 和平移向量 tvec
 * - 计算距离 distance 以及欧拉角 yaw / pitch / roll
 * - 支持云台坐标系与世界坐标系的转换
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "app/auto_aim/armor.hpp"

namespace app
{
	namespace auto_aim
	{

		/**
         * @brief Solver 初始化参数。
         *
         * 坐标系约定：
         * - 相机坐标系：OpenCV 约定，x 向右，y 向下，z 向前。
         * - 云台坐标系与世界坐标系由外参矩阵决定。
         * - 长度单位统一为 m。
         */
		struct SolverConfig
		{
			cv::Mat camera_matrix;
			cv::Mat distort_coeffs;

			Eigen::Matrix3d r_gimbal_to_imu_body = Eigen::Matrix3d::Identity();
			Eigen::Matrix3d r_camera_to_gimbal = Eigen::Matrix3d::Identity();
			Eigen::Vector3d t_camera_to_gimbal = Eigen::Vector3d::Zero();

			double lightbar_length_m = 56e-3;
			double small_armor_width_m = 135e-3;
			double big_armor_width_m = 230e-3;
		};

		class Solver
		{
		public:
			explicit Solver(const SolverConfig& config);

			/**
             * @brief 当前 Solver 配置是否有效。
            */
			bool is_valid() const noexcept;

			/**
            * @brief 获取云台坐标系到世界坐标系的旋转矩阵。
            */
			const Eigen::Matrix3d& r_gimbal_to_world() const noexcept;

			/**
            * @brief 根据 IMU 四元数更新云台到世界坐标系的旋转。
            * @param q_imu_body_to_world IMU body 坐标系到世界坐标系的旋转四元数。
            */
			void set_r_gimbal_to_world(const Eigen::Quaterniond& q_imu_body_to_world);

			/**
            * @brief 求解单块装甲板的三维位姿。
            * @param armor 输入四角点与类型，输出空间坐标和姿态。
            * @return true 求解成功；false 输入或配置非法、PnP 失败。
            */
			bool solve(Armor& armor) const;

		private:
			std::vector<cv::Point3d> object_points(ArmorType type) const;

		private:
			SolverConfig config_;

			cv::Mat camera_matrix_;
			cv::Mat distort_coeffs_;

			Eigen::Matrix3d r_gimbal_to_world_ = Eigen::Matrix3d::Identity();

			bool valid_ = false;
		};

	} // namespace auto_aim
} // namespace app

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP