#ifndef TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP
#define TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP

#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace tools
{
	namespace maths_tools
	{
		// 限制角度在 -π 到 π 之间
		double limit_rad(double angle);

		// 限制到 [min_value, max_value]
		double limit_min_max(double input, double min_value, double max_value);

		// 四元数转换为欧拉角（单位：弧度）
		// x = 0, y = 1, z = 2
		Eigen::Vector3d quat_to_euler(const Eigen::Quaterniond& quat, int axis0, int axis1,
		                              int axis2, bool is_extrinsic = false);

		// 旋转矩阵转换为欧拉角（单位：弧度）
		// x = 0, y = 1, z = 2
		Eigen::Vector3d rot_to_euler(const Eigen::Matrix3d& rot, int axis0, int axis1, int axis2,
		                             bool is_extrinsic = false);

		// 欧拉角转换为旋转矩阵（单位：弧度）
		// ypr: yaw pitch roll
		// zyx: 先绕 z 轴旋转，再绕 y 轴旋转，最后绕 x 轴旋转
		Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d& ypr);

		// 直角坐标系转换为球坐标系
		// 返回值为 yaw pitch distance
		Eigen::Vector3d cartesian_to_spherical(const Eigen::Vector3d& cartesian);

		// 球坐标系转换为直角坐标系
		// 输入值为 yaw pitch distance
		Eigen::Vector3d spherical_to_cartesian(const Eigen::Vector3d& spherical);

		// 直角坐标系转换为球坐标系的雅可比矩阵
		// 输入为 xyz，输出为 yaw pitch distance 对 xyz 的偏导
		Eigen::MatrixXd cartesian_to_spherical_jacobian(const Eigen::Vector3d& cartesian);

		// 球坐标系转换为直角坐标系的雅可比矩阵
		// 输入为 yaw pitch distance，输出为 xyz 对 yaw pitch distance 的偏导
		Eigen::MatrixXd spherical_to_cartesian_jacobian(const Eigen::Vector3d& spherical);

		// 计算两个时间点之间的时间差，单位：秒
		// 注意：返回值为 start - end
		double delta_time(const std::chrono::steady_clock::time_point& start,
		                  const std::chrono::steady_clock::time_point& end);

		// 计算两个二维向量之间夹角的绝对值，单位：弧度
		double get_abs_angle(const Eigen::Vector2d& vec1, const Eigen::Vector2d& vec2);

		// 平方函数
		template<typename T>
		T square(const T& value)
		{
			return value * value;
		}

		// 兼容旧 solver 命名：xyz -> yaw pitch distance
		Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

		// 兼容旧 solver 命名：yaw pitch distance -> xyz
		Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd);

		// 兼容旧 solver 命名：xyz -> yaw pitch distance 的雅可比矩阵
		Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d& xyz);

		// 兼容旧 solver 命名：yaw pitch distance -> xyz 的雅可比矩阵
		Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d& ypd);

	} // namespace maths_tools
} // namespace tools

#endif // TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP
