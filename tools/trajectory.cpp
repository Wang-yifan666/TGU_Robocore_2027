/**
 * @file trajectory.cpp
 * @brief 无空气阻力抛物线弹道求解实现。
 *
 * 迁移自 sp_vision_25/tools/trajectory.cpp，算法保持一致。
 */

#include "tools/trajectory.hpp"

#include <cmath>

namespace tools
{

	namespace
	{
		constexpr double kGravity = 9.7833; // m/s^2
	} // namespace

	Trajectory::Trajectory(double v0_mps, double d_m, double h_m)
	{
		// Safety hardening（不改变合法 SP25 输入的算法）：
		// 弹速非法 / 距离或高度非有限 / 负距离，直接判为无解，
		// 避免除零或 NaN 传播。
		if(!std::isfinite(v0_mps) || v0_mps <= 0.0 || !std::isfinite(d_m) || d_m < 0.0
		   || !std::isfinite(h_m))
		{
			unsolvable = true;
			return;
		}

		const double a = kGravity * d_m * d_m / (2.0 * v0_mps * v0_mps);
		const double b = -d_m;
		const double c = a + h_m;
		const double delta = b * b - 4.0 * a * c;

		if(delta < 0.0)
		{
			unsolvable = true;
			return;
		}

		unsolvable = false;

		const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2.0 * a);
		const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2.0 * a);

		const double pitch_1 = std::atan(tan_pitch_1);
		const double pitch_2 = std::atan(tan_pitch_2);

		const double fly_time_1 = d_m / (v0_mps * std::cos(pitch_1));
		const double fly_time_2 = d_m / (v0_mps * std::cos(pitch_2));

		// 取飞行时间较小的低弹道解。
		pitch = (fly_time_1 < fly_time_2) ? pitch_1 : pitch_2;
		fly_time = (fly_time_1 < fly_time_2) ? fly_time_1 : fly_time_2;
	}

} // namespace tools
