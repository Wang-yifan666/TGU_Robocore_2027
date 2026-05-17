#ifndef TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP
#define TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP

#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace tools::maths_tools
{
	// 限制角度在 -π 到 π 之间
	double limit_rad(double angle);
	// 限制到 [min_value, max_value]
	double limit_min_max(double input, double min_value, double max_value);

	// 四元数转换为欧拉角（单位：弧度）
	// x = 0, y = 1, z = 2
	Eigen::Vector3d quat_to_euler(Eigen::Quaterniond qaut, int axis0, int axis1, int axis2,
	                              bool is_extrinsic = false);

	// 旋转矩阵钻换为欧拉角（单位：弧度）
	// x = 0, y = 1, z = 2
	Eigen::Vector3d rot_to_euler(Eigen::Matrix3d rot, int axis0, int axis1, int axis2,
	                             bool is_extrinsic = false);


} // namespace tools::maths_tools

#endif // TGU_ROBOCORE_2027_TOOLS_MATHS_TOOLS_HPP