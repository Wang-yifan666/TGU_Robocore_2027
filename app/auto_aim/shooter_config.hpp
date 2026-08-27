/**
 * @file shooter_config.hpp
 * @brief Shooter 配置加载：TOML -> ShooterConfig。
 *
 * 职责：
 * - 所有生产字段显式 required，缺失即失败，不做 silent fallback；
 * - 校验数值 finite / 符号。
 */

#ifndef TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_CONFIG_HPP
#define TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_CONFIG_HPP

#include <string>

#include "app/auto_aim/shooter.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	/**
     * @brief 从已解析的 toml::table 加载 ShooterConfig（纯逻辑，便于单测）。
     * @param root 已解析的 TOML 根表，需含 [shooter] 表。
     * @param config 输出配置。
     * @return true 加载并校验成功；false 字段缺失或数值非法。
     */
	bool load_shooter_config_from_table(const toml::table& root, ShooterConfig& config);

	/**
     * @brief 从 TOML 文件加载 ShooterConfig。
     */
	bool load_shooter_config(const std::string& config_path, ShooterConfig& config);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_AUTO_AIM_SHOOTER_CONFIG_HPP
