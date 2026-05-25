#include "maths_tools.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace tools
{
	namespace maths_tools
	{
		double limit_rad(double angle)
		{
			constexpr double pi = std::numbers::pi;
			constexpr double two_pi = 2.0 * std::numbers::pi;

			angle = std::fmod(angle + pi, two_pi);

			if(angle <= 0.0)
			{
				angle += two_pi;
			}

			return angle - pi;
		}

		double limit_min_max(double input, double min_value, double max_value)
		{
			if(input > max_value)
			{
				return max_value;
			}

			if(input < min_value)
			{
				return min_value;
			}

			return input;
		}

		Eigen::Vector3d quat_to_euler(const Eigen::Quaterniond& quat, int axis0, int axis1,
		                              int axis2, bool is_extrinsic)
		{
			constexpr double pi = std::numbers::pi;
			constexpr double eps = 1e-7;

			if(!is_extrinsic)
			{
				std::swap(axis0, axis2);
			}

			int i = axis0;
			const int j = axis1;
			int k = axis2;

			const bool is_proper = (i == k);

			if(is_proper)
			{
				k = 3 - i - j;
			}

			const int sign = (i - j) * (j - k) * (k - i) / 2;

			double a = 0.0;
			double b = 0.0;
			double c = 0.0;
			double d = 0.0;

			const Eigen::Vector4d xyzw = quat.coeffs();

			if(is_proper)
			{
				a = xyzw[3];
				b = xyzw[i];
				c = xyzw[j];
				d = xyzw[k] * sign;
			}
			else
			{
				a = xyzw[3] - xyzw[j];
				b = xyzw[i] + xyzw[k] * sign;
				c = xyzw[j] + xyzw[3];
				d = xyzw[k] * sign - xyzw[i];
			}

			const double n2 = a * a + b * b + c * c + d * d;

			if(n2 <= eps)
			{
				return Eigen::Vector3d::Zero();
			}

			Eigen::Vector3d euler = Eigen::Vector3d::Zero();

			double acos_value = 2.0 * (a * a + b * b) / n2 - 1.0;
			acos_value = std::clamp(acos_value, -1.0, 1.0);

			euler[1] = std::acos(acos_value);

			const double half_sum = std::atan2(b, a);
			const double half_diff = std::atan2(-d, c);

			const bool safe1 = std::abs(euler[1]) >= eps;
			const bool safe2 = std::abs(euler[1] - pi) >= eps;
			const bool safe = safe1 && safe2;

			if(safe)
			{
				euler[0] = half_sum + half_diff;
				euler[2] = half_sum - half_diff;
			}
			else // 处理万向锁
			{
				if(!is_extrinsic)
				{
					euler[0] = 0.0;

					if(!safe1)
					{
						euler[2] = 2.0 * half_sum;
					}

					if(!safe2)
					{
						euler[2] = -2.0 * half_diff;
					}
				}
				else
				{
					euler[2] = 0.0;

					if(!safe1)
					{
						euler[0] = 2.0 * half_sum;
					}

					if(!safe2)
					{
						euler[0] = 2.0 * half_diff;
					}
				}
			}

			for(int index = 0; index < 3; ++index)
			{
				euler[index] = limit_rad(euler[index]);
			}

			if(!is_proper)
			{
				euler[2] *= sign;
				euler[1] -= pi / 2.0;
			}

			if(!is_extrinsic)
			{
				std::swap(euler[0], euler[2]);
			}

			return euler;
		}

		Eigen::Vector3d rot_to_euler(const Eigen::Matrix3d& rot, int axis0, int axis1, int axis2,
		                             bool is_extrinsic)
		{
			const Eigen::Quaterniond quat(rot);
			return quat_to_euler(quat, axis0, axis1, axis2, is_extrinsic);
		}

		Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d& ypr)
		{
			const double yaw = ypr[0];
			const double pitch = ypr[1];
			const double roll = ypr[2];

			const double cos_yaw = std::cos(yaw);
			const double sin_yaw = std::sin(yaw);
			const double cos_pitch = std::cos(pitch);
			const double sin_pitch = std::sin(pitch);
			const double cos_roll = std::cos(roll);
			const double sin_roll = std::sin(roll);

			Eigen::Matrix3d rot;

			// clang-format off
			rot << cos_yaw * cos_pitch,
			       cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
			       cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,

			       sin_yaw * cos_pitch,
			       sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
			       sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,

			      -sin_pitch,
			       cos_pitch * sin_roll,
			       cos_pitch * cos_roll;
			// clang-format on

			return rot;
		}

		Eigen::Vector3d cartesian_to_spherical(const Eigen::Vector3d& cartesian)
		{
			const double x = cartesian[0];
			const double y = cartesian[1];
			const double z = cartesian[2];

			const double yaw = std::atan2(y, x);
			const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
			const double distance = std::sqrt(x * x + y * y + z * z);

			return Eigen::Vector3d{yaw, pitch, distance};
		}

		Eigen::Vector3d spherical_to_cartesian(const Eigen::Vector3d& spherical)
		{
			const double yaw = spherical[0];
			const double pitch = spherical[1];
			const double distance = spherical[2];

			const double x = distance * std::cos(pitch) * std::cos(yaw);
			const double y = distance * std::cos(pitch) * std::sin(yaw);
			const double z = distance * std::sin(pitch);

			return Eigen::Vector3d{x, y, z};
		}

		Eigen::MatrixXd cartesian_to_spherical_jacobian(const Eigen::Vector3d& cartesian)
		{
			constexpr double eps = 1e-9;

			const double x = cartesian[0];
			const double y = cartesian[1];
			const double z = cartesian[2];

			const double xy_square = x * x + y * y;
			const double xy_distance = std::sqrt(xy_square);
			const double distance_square = xy_square + z * z;
			const double distance = std::sqrt(distance_square);

			Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 3);

			if(xy_square > eps)
			{
				jacobian(0, 0) = -y / xy_square;
				jacobian(0, 1) = x / xy_square;
				jacobian(0, 2) = 0.0;
			}

			if(xy_distance > eps && distance_square > eps)
			{
				jacobian(1, 0) = -x * z / (xy_distance * distance_square);
				jacobian(1, 1) = -y * z / (xy_distance * distance_square);
				jacobian(1, 2) = xy_distance / distance_square;
			}

			if(distance > eps)
			{
				jacobian(2, 0) = x / distance;
				jacobian(2, 1) = y / distance;
				jacobian(2, 2) = z / distance;
			}

			return jacobian;
		}

		Eigen::MatrixXd spherical_to_cartesian_jacobian(const Eigen::Vector3d& spherical)
		{
			const double yaw = spherical[0];
			const double pitch = spherical[1];
			const double distance = spherical[2];

			const double cos_yaw = std::cos(yaw);
			const double sin_yaw = std::sin(yaw);
			const double cos_pitch = std::cos(pitch);
			const double sin_pitch = std::sin(pitch);

			Eigen::MatrixXd jacobian(3, 3);

			// clang-format off
			jacobian << -distance * cos_pitch * sin_yaw,
			            -distance * sin_pitch * cos_yaw,
			             cos_pitch * cos_yaw,

			             distance * cos_pitch * cos_yaw,
			            -distance * sin_pitch * sin_yaw,
			             cos_pitch * sin_yaw,

			             0.0,
			             distance * cos_pitch,
			             sin_pitch;
			// clang-format on

			return jacobian;
		}

		double delta_time(const std::chrono::steady_clock::time_point& start,
		                  const std::chrono::steady_clock::time_point& end)
		{
			const std::chrono::duration<double> duration = start - end;
			return duration.count();
		}

		double get_abs_angle(const Eigen::Vector2d& vec1, const Eigen::Vector2d& vec2)
		{
			constexpr double eps = 1e-9;

			const double norm1 = vec1.norm();
			const double norm2 = vec2.norm();

			if(norm1 <= eps || norm2 <= eps)
			{
				return 0.0;
			}

			double cos_theta = vec1.dot(vec2) / (norm1 * norm2);
			cos_theta = std::clamp(cos_theta, -1.0, 1.0);

			return std::acos(cos_theta);
		}

		Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz)
		{
			return cartesian_to_spherical(xyz);
		}

		Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd)
		{
			return spherical_to_cartesian(ypd);
		}

		Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d& xyz)
		{
			return cartesian_to_spherical_jacobian(xyz);
		}

		Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d& ypd)
		{
			return spherical_to_cartesian_jacobian(ypd);
		}

	} // namespace maths_tools
} // namespace tools
