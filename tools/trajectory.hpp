/**
 * @file trajectory.hpp
 * @brief 无空气阻力抛物线弹道求解。
 *
 * 迁移自 sp_vision_25/tools/trajectory.hpp，保留其弹道语义：
 * - 不考虑空气阻力；
 * - 重力 g = 9.7833 m/s^2；
 * - 取飞行时间较小的低弹道解；
 * - pitch 抬头为正（rad）。
 */

#ifndef TGU_ROBOCORE_2027_TOOLS_TRAJECTORY_HPP
#define TGU_ROBOCORE_2027_TOOLS_TRAJECTORY_HPP

namespace tools
{

	/**
	 * @brief 无空气阻力弹道求解结果。
	 *
	 * 单位约定：
	 * - 长度 m；
	 * - 速度 m/s；
	 * - 时间 s；
	 * - 角度 rad。
	 */
	struct Trajectory
	{
		bool unsolvable = false; ///< 弹道无解（delta < 0 或输入非法）。
		double fly_time = 0.0;   ///< 弹丸飞行时间（s）。
		double pitch = 0.0;      ///< 发射俯仰角（rad），抬头为正。

		/**
		 * @brief 求解无空气阻力弹道。
		 * @param v0_mps 弹丸初速度大小（m/s），必须有限且 > 0。
		 * @param d_m 目标水平距离（m），必须有限且 >= 0。
		 * @param h_m 目标竖直高度（m），必须有限。
		 *
		 * 输入非法（v0 <= 0 / NaN / Inf，或 d/h 非有限、d < 0）时
		 * 置 unsolvable = true，fly_time / pitch 保持 0。
		 */
		Trajectory(double v0_mps, double d_m, double h_m);
	};

} // namespace tools

#endif // TGU_ROBOCORE_2027_TOOLS_TRAJECTORY_HPP
