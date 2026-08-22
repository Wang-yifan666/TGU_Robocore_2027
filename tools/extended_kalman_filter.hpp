/**
 * @file extended_kalman_filter.hpp
 * @brief 通用、数值稳定、与 RoboMaster 业务无关的扩展卡尔曼滤波器。
 *
 * 仅依赖 Eigen，不感知任何业务语义（Target / Armor / Tracker / Detector /
 * Solver / 弹道 均与本类无关）。
 *
 * 数学模型（n = state 维度，m = measurement 维度）：
 *
 *   Linear predict:
 *       x = F x
 *       P = F P F^T + Q
 *
 *   Nonlinear predict:
 *       x = f(x)
 *       P = F P F^T + Q
 *       其中 F = df/dx evaluated at x_prior。
 *
 *   Update（严格 prior -> posterior 顺序）:
 *       z_pred     = h(x_prior)              // 线性时为 H * x_prior
 *       innovation = residual(z, z_pred)
 *       S          = H P_prior H^T + R
 *       K          = P_prior H^T S^-1
 *       x_post     = state_add(x_prior, K innovation)
 *       A          = I - K H
 *       P_post     = A P_prior A^T + K R K^T   // Joseph form
 *
 *   NIS（标准先验，仅作诊断）:
 *       NIS = innovation^T S^-1 innovation
 *
 * 注意：
 * - 不显式求 S/P 的逆，使用 Eigen::LDLT 的 solve。
 * - P0 / Q / R 在数学语义上应为 positive-semidefinite 协方差矩阵。
 *   本实现主动检查 shape / finite / symmetry；PSD 仍属 caller precondition。
 */

#ifndef TGU_ROBOCORE_2027_TOOLS_EXTENDED_KALMAN_FILTER_HPP
#define TGU_ROBOCORE_2027_TOOLS_EXTENDED_KALMAN_FILTER_HPP

#include <cstddef>
#include <functional>
#include <limits>

#include <Eigen/Dense>

namespace tools
{

	/**
	 * @brief 通用扩展卡尔曼滤波器。
	 *
	 * 角度周期性等业务语义由调用方通过 `state_add` / `residual` hook 表达，
	 * 本类不包含任何与具体测量分量含义相关的逻辑。
	 */
	class ExtendedKalmanFilter
	{
	public:
		/**
		 * @brief 状态转移函数 f(x)，返回同维度 state。
		 */
		using StateFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

		/**
		 * @brief 自定义状态加法：a + b，默认向量加。
		 */
		using StateAddFn =
		    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

		/**
		 * @brief 自定义测量残差：residual(a, b)，默认 a - b。
		 */
		using ResidualFn =
		    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

		/**
		 * @brief 初始化状态与协方差，并绑定状态加法与残差 hook。
		 *
		 * @throw std::invalid_argument x0/P0 维度非法、非有限或 P0 非近似对称；
		 *        或传入空的 state_add / residual。
		 */
		ExtendedKalmanFilter(
		    const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
		    StateAddFn state_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			    return a + b;
		    },
		    ResidualFn residual = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			    return a - b;
		    });

		/**
		 * @brief 重新初始化状态与协方差，并清空 diagnostics。
		 *
		 * @throw std::invalid_argument 同构造函数。
		 */
		void reset(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);

		/**
		 * @brief 线性预测：x = F x；P = F P F^T + Q。
		 *
		 * @throw std::invalid_argument F/Q 维度非法、非有限或 Q 非近似对称。
		 * @throw std::runtime_error 合法输入下预测产生非有限结果。
		 * 任何异常下 state/covariance 保持不变。
		 */
		void predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

		/**
		 * @brief 非线性预测：x = f(x)；P = F P F^T + Q，
		 *        其中 F = df/dx evaluated at x_prior。
		 *
		 * @throw std::invalid_argument F/Q/f() 维度非法、非有限、
		 *        Q 非近似对称或 f() 返回非法（维度/有限性）。
		 * @throw std::runtime_error 合法输入下预测产生非有限结果。
		 * 任何异常下 state/covariance 保持不变。
		 */
		void predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q, const StateFn& f);

		/**
		 * @brief 线性测量更新：h(x) = H x。
		 *
		 * @return false 仅表示数值失败（LDLT 分解/solve 失败或结果非有限），
		 *              此时 state/covariance/diagnostics 全部保持不变。
		 * @throw std::invalid_argument z/H/R 维度非法、非有限或 R 非近似对称。
		 */
		bool update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);

		/**
		 * @brief 非线性测量更新：使用提供的 h(x)。
		 *
		 * H = dh/dx evaluated at x_prior；h() 用 x_prior 求值。
		 *
		 * @return false 仅表示数值失败，成员保持不变。
		 * @throw std::invalid_argument z/H/R/h()/residual()/state_add() 维度非法、
		 *        非有限或 R 非近似对称。
		 */
		bool update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
		            const StateFn& h);

		/**
		 * @brief 当前后验状态。
		 */
		const Eigen::VectorXd& state() const noexcept;

		/**
		 * @brief 当前后验协方差。
		 */
		const Eigen::MatrixXd& covariance() const noexcept;

		/**
		 * @brief state 维度 n。
		 */
		std::size_t state_dim() const noexcept;

		/**
		 * @brief 最近一次成功 measurement update 的 prior innovation。
		 *
		 * 语义：上一次成功 update 中、posterior 应用之前计算的
		 * residual(z, h(x_prior))。
		 * 未发生过成功 update（构造/reset/失败）时为空 vector。
		 */
		const Eigen::VectorXd& last_innovation() const noexcept;

		/**
		 * @brief 最近一次成功 measurement update 的标准 prior NIS。
		 *
		 * 仅作诊断。注意：update() 成功返回时 posterior 已应用，
		 * 因此该值不可用于事后拒绝本次 measurement 的 gating。
		 * 未来若需 pre-update NIS gating，将另行设计 innovation-statistics API。
		 *
		 * 构造/reset 后、或尚无成功 update 时为 NaN。该数据可能 stale
		 * （predict 不会刷新它）。
		 */
		double last_nis() const noexcept;

	private:
		bool update_impl(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
		                 const Eigen::MatrixXd& R, const StateFn& h);

		// 共享 predict 尾部：计算 F P F^T + Q、验有限、对称化、commit x_next/P_。
		// 前提：F/Q/x_next 均已通过前置校验。
		void predict_commit(Eigen::VectorXd x_next, const Eigen::MatrixXd& F,
		                    const Eigen::MatrixXd& Q);

		Eigen::VectorXd x_;
		Eigen::MatrixXd P_;

		StateAddFn state_add_;
		ResidualFn residual_;

		Eigen::VectorXd last_innovation_;
		double last_nis_ = std::numeric_limits<double>::quiet_NaN();
	};

} // namespace tools

#endif // TGU_ROBOCORE_2027_TOOLS_EXTENDED_KALMAN_FILTER_HPP