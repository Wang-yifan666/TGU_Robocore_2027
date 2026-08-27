/**
 * @file shooter.hpp
 * @brief Shooter：轻量开火决策模块（只回答本帧是否允许开火）。
 *
 * 职责边界：
 * - 输出只允许 bool（允许/禁止开火），不开火原因通过 Logger 记录；
 * - 不依赖 io / Tracker / Aimer 对象 / debug state；
 * - 跨帧只保存“上一帧有效 aiming yaw”。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_HPP

#include <optional>

#include "app/auto_aim/aimer.hpp"

namespace app::auto_aim
{

	/**
     * @brief Shooter 配置（角度单位 rad、距离单位 m）。
     */
	struct ShooterConfig
	{
		/// 自动开火开关。程序化默认 false（fail-safe）；生产 TOML 显式开启。
		bool auto_fire = false;

		double near_tolerance_rad = 0.0;   ///< 近距离容差（较松）。
		double far_tolerance_rad = 0.0;    ///< 远距离容差（较严）。
		double distance_threshold_m = 0.0; ///< 远近阈值（m）。
	};

	/**
     * @brief 校验 ShooterConfig；非法时抛 std::invalid_argument。
     */
	void validate_shooter_config(const ShooterConfig& config);

	/**
     * @brief 构造程序化默认配置（测试/演示用，非生产 silent fallback）。
     */
	ShooterConfig make_default_shooter_config();

	/**
     * @brief Shooter：从 AimingSolution 与标量输入决定本帧是否允许开火。
     */
	class Shooter
	{
	public:
		explicit Shooter(const ShooterConfig& config);

		/**
         * @brief 本帧开火决策。
         *
         * @param aiming Aimer 输出（只消费 valid / yaw_rad / pitch_rad）。
         * @param target_distance_m 目标车辆中心水平距离（m，= hypot(x,y)）。
         * @param gimbal_yaw_rad 当前实测云台 yaw（与 aiming.yaw_rad 同坐标系）。
         * @return true 允许开火。
         */
		bool shoot(const AimingSolution& aiming, double target_distance_m, double gimbal_yaw_rad);

		/**
         * @brief 清空跨帧状态（上一帧有效 aiming yaw）。
         */
		void reset();

	private:
		ShooterConfig config_;

		std::optional<double> previous_aiming_yaw_rad_;
	};

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_HPP
