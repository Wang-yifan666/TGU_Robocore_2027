/**
 * @file solver_config.hpp
 * @brief Solver 配置加载：TOML → SolverConfig。
 *
 * 职责：
 * - 检查 matrix/vector 元素数量并校验数值 finite；
 * - 对缺失或非法字段返回失败，不静默 fallback 到虚假的相机内参；
 */

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_CONFIG_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_CONFIG_HPP

#include <string>

#include "app/auto_aim/solver.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	/**
	 * @brief 从已解析的 toml::table 加载 SolverConfig（纯逻辑，便于单测）。
	 * @param root 已解析的 TOML 根表。
	 * @param config 输出配置。
	 * @return true 加载并校验成功；false 字段缺失或数值非法。
	 *
	 * 配置约定：
	 * - 矩阵按行主序（row-major）填写，共 9 个元素；
	 * - 平移向量按 [tx, ty, tz]；
	 * - 长度单位统一为 m；
	 * - armor 尺寸（lightbar_length_m 等）不在此加载，保留 SolverConfig 默认值。
	 */
	bool load_solver_config_from_table(const toml::table& root, SolverConfig& config);

	/**
	 * @brief 从 TOML 文件加载 SolverConfig。
	 * @param config_path TOML 文件路径。
	 * @param config 输出配置。
	 * @return true 加载并校验成功；false 解析失败、字段缺失或数值非法。
	 */
	bool load_solver_config(const std::string& config_path, SolverConfig& config);

} // namespace app::auto_aim

#endif // TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_CONFIG_HPP