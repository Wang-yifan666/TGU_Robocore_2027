#include "tools/extended_kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tools
{

	namespace
	{

		constexpr double kSymmetryTolerance = 1e-9;

		// solve 结果的向后误差相对容差。仅用于判定 "S x = rhs 是否真的被解出"，
		// 不是 condition-number threshold，也不是 regularization / jitter。
		constexpr double kSolveRelativeTolerance = 1e-9;

		bool all_finite(const Eigen::VectorXd& vector)
		{
			return vector.allFinite();
		}

		bool all_finite(const Eigen::MatrixXd& matrix)
		{
			return matrix.allFinite();
		}

		bool is_square(const Eigen::MatrixXd& matrix, Eigen::Index expected)
		{
			return matrix.rows() == expected && matrix.cols() == expected;
		}

		// 近似对称：按幅值缩放 tolerance，不要求 bitwise 对称。
		bool approximately_symmetric(const Eigen::MatrixXd& matrix)
		{
			if(!all_finite(matrix))
			{
				return false;
			}

			const double scale = matrix.norm();
			const double tolerance = kSymmetryTolerance * std::max(1.0, scale);

			return (matrix - matrix.transpose()).norm() <= tolerance;
		}

		void validate_state(const Eigen::VectorXd& x, Eigen::Index expected_dim)
		{
			if(x.size() == 0 || x.rows() != expected_dim || x.cols() != 1 || !all_finite(x))
			{
				throw std::invalid_argument("state vector must be (n x 1) and finite");
			}
		}

		void validate_covariance(const Eigen::MatrixXd& p, Eigen::Index expected_dim,
		                         const char* name)
		{
			if(!is_square(p, expected_dim) || !all_finite(p) || !approximately_symmetric(p))
			{
				throw std::invalid_argument(std::string(name)
				                            + " must be square, finite, "
				                              "and approximately symmetric");
			}
		}

		Eigen::MatrixXd symmetrize(const Eigen::MatrixXd& matrix)
		{
			// 使用 .eval() 避免 expression aliasing。
			return (0.5 * (matrix + matrix.transpose())).eval();
		}

		// 用 LDLT 解 S x = rhs，并做向后误差校验。
		// 仅凭 info()==Success 无法识别全部奇异/病态 S（例如 1x1 零矩阵），
		// 因此额外校验 ||S*out - rhs|| <= tol * max(1, ||rhs||)。
		bool solve_ldlt(const Eigen::LDLT<Eigen::MatrixXd>& ldlt, const Eigen::MatrixXd& S,
		                const Eigen::MatrixXd& rhs, Eigen::MatrixXd& out)
		{
			if(ldlt.info() != Eigen::Success)
			{
				return false;
			}

			out = ldlt.solve(rhs);

			if(!out.allFinite())
			{
				return false;
			}

			const double residual = (S * out - rhs).norm();
			const double scale = std::max(1.0, rhs.norm());

			return residual <= kSolveRelativeTolerance * scale;
		}

	} // namespace

	ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
	                                           StateAddFn state_add, ResidualFn residual):
	state_add_(std::move(state_add)), residual_(std::move(residual))
	{
		if(!state_add_)
		{
			throw std::invalid_argument("state_add must be callable");
		}

		if(!residual_)
		{
			throw std::invalid_argument("residual must be callable");
		}

		reset(x0, P0);
	}

	void ExtendedKalmanFilter::reset(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
	{
		if(x0.size() == 0 || x0.cols() != 1 || !all_finite(x0))
		{
			throw std::invalid_argument("x0 must be (n x 1) and finite");
		}

		if(!is_square(P0, x0.rows()) || !all_finite(P0) || !approximately_symmetric(P0))
		{
			throw std::invalid_argument("P0 must be square, finite, and approximately symmetric");
		}

		x_ = x0;
		P_ = P0;
		last_innovation_ = Eigen::VectorXd();
		last_nis_ = std::numeric_limits<double>::quiet_NaN();
	}

	void ExtendedKalmanFilter::predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q)
	{
		const Eigen::Index n = x_.size();

		if(!is_square(F, n) || !all_finite(F))
		{
			throw std::invalid_argument("F must be (n x n) and finite");
		}

		validate_covariance(Q, n, "Q");

		// 内部线性算术 F*x。若因 overflow 产生 non-finite，
		// 属 numerical failure，抛 runtime_error 且状态/协方差不变。
		Eigen::VectorXd x_next = F * x_;

		if(!all_finite(x_next))
		{
			throw std::runtime_error("linear predict produced non-finite state");
		}

		predict_commit(std::move(x_next), F, Q);
	}

	void ExtendedKalmanFilter::predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
	                                   const StateFn& f)
	{
		const Eigen::Index n = x_.size();

		if(!is_square(F, n) || !all_finite(F))
		{
			throw std::invalid_argument("F must be (n x n) and finite");
		}

		validate_covariance(Q, n, "Q");

		if(!f)
		{
			throw std::invalid_argument("f must be callable");
		}

		// 在 x_prior 处求值 f。若 f() 返回维度非法/NaN/Inf，
		// 属 callback contract error，抛 invalid_argument。
		Eigen::VectorXd x_next = f(x_);
		validate_state(x_next, n);

		predict_commit(std::move(x_next), F, Q);
	}

	void ExtendedKalmanFilter::predict_commit(Eigen::VectorXd x_next, const Eigen::MatrixXd& F,
	                                          const Eigen::MatrixXd& Q)
	{
		Eigen::MatrixXd p_next = F * P_ * F.transpose() + Q;

		if(!all_finite(p_next))
		{
			throw std::runtime_error("predict produced non-finite covariance");
		}

		p_next = symmetrize(p_next);

		// 全部数学完成后再 commit。
		x_ = std::move(x_next);
		P_ = std::move(p_next);
	}

	bool ExtendedKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
	                                  const Eigen::MatrixXd& R)
	{
		return update_impl(z, H, R, [&H](const Eigen::VectorXd& x) {
			return H * x;
		});
	}

	bool ExtendedKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
	                                  const Eigen::MatrixXd& R, const StateFn& h)
	{
		if(!h)
		{
			throw std::invalid_argument("h must be callable");
		}

		return update_impl(z, H, R, h);
	}

	bool ExtendedKalmanFilter::update_impl(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
	                                       const Eigen::MatrixXd& R, const StateFn& h)
	{
		const Eigen::Index n = x_.size();

		if(z.size() == 0 || z.cols() != 1 || !all_finite(z))
		{
			throw std::invalid_argument("z must be (m x 1) and finite");
		}

		const Eigen::Index m = z.rows();

		if(H.rows() != m || H.cols() != n || !all_finite(H))
		{
			throw std::invalid_argument("H must be (m x n) and finite");
		}

		validate_covariance(R, m, "R");

		// ---- 全部使用 prior 状态计算 ----
		const Eigen::VectorXd x_prior = x_;
		const Eigen::MatrixXd p_prior = P_;

		Eigen::VectorXd z_pred = h(x_prior);
		validate_state(z_pred, m);

		Eigen::VectorXd innovation = residual_(z, z_pred);
		validate_state(innovation, m);

		Eigen::MatrixXd S = H * p_prior * H.transpose() + R;

		Eigen::LDLT<Eigen::MatrixXd> ldlt(S);

		if(ldlt.info() != Eigen::Success)
		{
			// 数值失败：状态/协方差/diagnostics 全部保持不变。
			return false;
		}

		// K^T = S^{-1} H P_prior，通过 solve 得到，不显式求逆。
		// solve_ldlt 做向后误差校验，避免把奇异/病态 S 当作成功。
		Eigen::MatrixXd K_transpose;
		if(!solve_ldlt(ldlt, S, (H * p_prior).eval(), K_transpose))
		{
			return false;
		}

		const Eigen::MatrixXd K = K_transpose.transpose();

		// 标准 prior NIS（仅诊断）：NIS = innovation^T S^{-1} innovation。
		const Eigen::MatrixXd innovation_matrix = innovation;
		Eigen::MatrixXd innovation_normalized;
		if(!solve_ldlt(ldlt, S, innovation_matrix, innovation_normalized))
		{
			return false;
		}

		const double nis = (innovation.transpose() * innovation_normalized)(0, 0);

		if(!std::isfinite(nis))
		{
			return false;
		}

		// ---- 计算 posterior ----
		// correction = K * innovation 属内部数值运算。
		// 若 non-finite，是 numerical failure（非 callback contract error），
		// 返回 false 并完整 rollback。
		Eigen::VectorXd correction = K * innovation;

		if(!all_finite(correction))
		{
			return false;
		}

		Eigen::VectorXd x_post = state_add_(x_prior, correction);
		validate_state(x_post, n);

		const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
		const Eigen::MatrixXd A = I - K * H;

		Eigen::MatrixXd p_post = A * p_prior * A.transpose() + K * R * K.transpose();

		if(!all_finite(x_post) || !all_finite(p_post))
		{
			return false;
		}

		p_post = symmetrize(p_post);

		// 全部数学完成后再 commit。
		x_ = std::move(x_post);
		P_ = std::move(p_post);
		last_innovation_ = std::move(innovation);
		last_nis_ = nis;

		return true;
	}

	const Eigen::VectorXd& ExtendedKalmanFilter::state() const noexcept
	{
		return x_;
	}

	const Eigen::MatrixXd& ExtendedKalmanFilter::covariance() const noexcept
	{
		return P_;
	}

	std::size_t ExtendedKalmanFilter::state_dim() const noexcept
	{
		return static_cast<std::size_t>(x_.size());
	}

	const Eigen::VectorXd& ExtendedKalmanFilter::last_innovation() const noexcept
	{
		return last_innovation_;
	}

	double ExtendedKalmanFilter::last_nis() const noexcept
	{
		return last_nis_;
	}

} // namespace tools