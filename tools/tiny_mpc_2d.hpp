/**
 * @file tiny_mpc_2d.hpp
 * @brief TinyMPC 的 Robocore 自研封装：2 状态 / 1 输入双积分器 MPC。
 *
 * 第三方 TinyMPC vendor 源码位于 tools/tinympc/（保持原样，见其 README）。
 * 本文件只做类型安全、RAII、确定性冷启动的适配，不修改 vendor 实现。
 *
 * 边界：本封装只依赖 Eigen（经 vendor）。禁止依赖 app / io / task / TOML。
 */

#ifndef TGU_ROBOCORE_2027_TOOLS_TINY_MPC_2D_HPP
#define TGU_ROBOCORE_2027_TOOLS_TINY_MPC_2D_HPP

#include <Eigen/Core>

#include <memory>

namespace tools
{

	/**
	 * @brief 2 状态（位置/速度）、1 输入（加速度）的线性 MPC 求解器封装。
	 *
	 * 模型：A = [[1, dt], [0, 1]]，B = [[0], [dt]]，f = 0。
	 * 约束：状态无界，|u| <= max_acceleration。
	 *
	 * 每次 solve() 前恢复到 setup 后的冷启动快照，保证不同历史调用之间
	 * 数值确定（deterministic cold start）；adaptive rho 保持禁用。
	 */
	class TinyMpc2d
	{
	public:
		struct Config
		{
			double dt = 0.01;                          ///< 离散时间步长（s）。
			double rho = 1.0;                          ///< ADMM 增广拉格朗日惩罚（> 0）。
			Eigen::Vector2d q = Eigen::Vector2d(9e6, 0.0); ///< 状态权重（允许半正定）。
			double r = 1.0;                            ///< 输入权重（> 0）。
			double max_acceleration = 50.0;            ///< 加速度幅值上限（> 0）。
			int horizon = 100;                         ///< MPC 时域长度（>= 2）。
			int max_iter = 10;                         ///< ADMM 最大迭代（>= 1）。
		};

		/**
		 * @brief 构造并配置 solver；非法 config 抛 std::invalid_argument。
		 */
		explicit TinyMpc2d(const Config& config);

		~TinyMpc2d();

		TinyMpc2d(const TinyMpc2d&) = delete;
		TinyMpc2d& operator=(const TinyMpc2d&) = delete;
		TinyMpc2d(TinyMpc2d&& other) noexcept;
		TinyMpc2d& operator=(TinyMpc2d&& other) noexcept;

		/**
		 * @brief 从 x0 出发跟踪参考轨迹 x_ref（2 x horizon）求解。
		 * @param x0 初始状态 [position, velocity]。
		 * @param x_ref 参考轨迹，行 0 = position、行 1 = velocity。
		 * @return 0 表示 tiny_solve 收敛；1 表示达到 max_iter 仍未收敛；
		 *         负值表示输入非法或 solver 未就绪。
		 */
		int solve(const Eigen::Vector2d& x0,
		          const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref);

		double position(int k) const;     ///< 状态 0（位置）在时域下标 k 的值（primal work->x）。
		double velocity(int k) const;     ///< 状态 1（速度）在时域下标 k 的值（primal work->x）。
		double acceleration(int k) const; ///< 输入在 k 的值（projected solution->u，有界 |acc| <= max_acc）。
		double primal_acceleration(int k) const; ///< 输入在 k 的值（primal work->u，可能越界）。

		bool solved() const;              ///< ADMM 是否满足 termination condition。
		int iteration_count() const;      ///< 本次 solve 的实际迭代次数。
		double input_primal_residual() const; ///< 输入 primal residual（终止检查时的值）。

		int horizon() const noexcept;

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

} // namespace tools

#endif // TGU_ROBOCORE_2027_TOOLS_TINY_MPC_2D_HPP
