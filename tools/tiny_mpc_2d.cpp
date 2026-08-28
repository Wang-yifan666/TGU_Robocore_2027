/**
 * @file tiny_mpc_2d.cpp
 * @brief TinyMpc2d 实现（pimpl，隔离 vendor 头与 using namespace Eigen）。
 */

#include "tools/tiny_mpc_2d.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "tools/tinympc/tiny_api.hpp"

namespace tools
{

	namespace
	{

		void require(bool condition, const char* message)
		{
			if(!condition)
			{
				throw std::invalid_argument(std::string("TinyMpc2d: ") + message);
			}
		}

	} // namespace

	struct TinyMpc2d::Impl
	{
		TinySolver* solver = nullptr;
		int horizon = 0;

		// setup 后的冷启动快照（深拷贝）。每次 solve 前恢复，保证 deterministic cold start。
		std::unique_ptr<TinySolution> cold_solution;
		std::unique_ptr<TinyWorkspace> cold_work;

		~Impl()
		{
			release();
		}

		// legacy tiny_setup 内部 new 了 solution/cache/settings/work/solver，
		// 且没有 cleanup API；本封装按所有权逆序释放，避免泄漏 / double-free。
		void release()
		{
			if(solver != nullptr)
			{
				delete solver->work;
				delete solver->cache;
				delete solver->settings;
				delete solver->solution;
				delete solver;
				solver = nullptr;
			}
		}

		void save_cold_snapshot()
		{
			cold_solution.reset(new TinySolution());
			*cold_solution = *solver->solution;

			cold_work.reset(new TinyWorkspace());
			*cold_work = *solver->work;
		}

		void restore_cold_snapshot()
		{
			*solver->solution = *cold_solution;
			*solver->work = *cold_work;
		}
	};

	TinyMpc2d::TinyMpc2d(const Config& config): impl_(std::make_unique<Impl>())
	{
		require(std::isfinite(config.dt) && config.dt > 0.0, "dt must be finite and > 0");
		require(std::isfinite(config.rho) && config.rho > 0.0, "rho must be finite and > 0");
		require(config.q.allFinite() && config.q.minCoeff() >= 0.0, "q must be finite and >= 0");
		require(std::isfinite(config.r) && config.r > 0.0, "r must be finite and > 0");
		require(std::isfinite(config.max_acceleration) && config.max_acceleration > 0.0,
		        "max_acceleration must be finite and > 0");
		require(config.horizon >= 2, "horizon must be >= 2");
		require(config.max_iter >= 1, "max_iter must be >= 1");

		Eigen::MatrixXd A(2, 2);
		A << 1.0, config.dt, 0.0, 1.0;

		Eigen::MatrixXd B(2, 1);
		B << 0.0, config.dt;

		Eigen::VectorXd f = Eigen::VectorXd::Zero(2);

		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
		Q(0, 0) = config.q(0);
		Q(1, 1) = config.q(1);

		Eigen::MatrixXd R(1, 1);
		R(0, 0) = config.r;

		TinySolver* solver = nullptr;
		const int setup_status = tiny_setup(&solver, A, B, f, Q, R, config.rho, 2, 1,
		                                    config.horizon, 0);
		if(setup_status != 0 || solver == nullptr)
		{
			// tiny_setup 内部已 new 全部子对象；失败时由本封装接管释放。
			if(solver != nullptr)
			{
				delete solver->work;
				delete solver->cache;
				delete solver->settings;
				delete solver->solution;
				delete solver;
			}
			throw std::invalid_argument("TinyMpc2d: tiny_setup failed");
		}

		impl_->solver = solver;
		impl_->horizon = config.horizon;

		// 状态无界，仅限制输入（加速度）幅值。
		Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, config.horizon, -1e17);
		Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, config.horizon, 1e17);
		Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, config.horizon - 1,
		                                                  -config.max_acceleration);
		Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, config.horizon - 1,
		                                                  config.max_acceleration);
		tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

		// 静态配置：迭代上限 + 显式禁用 adaptive rho（本轮不支持）。
		solver->settings->max_iter = config.max_iter;
		solver->settings->adaptive_rho = 0;

		impl_->save_cold_snapshot();
	}

	TinyMpc2d::~TinyMpc2d() = default;

	TinyMpc2d::TinyMpc2d(TinyMpc2d&& other) noexcept = default;
	TinyMpc2d& TinyMpc2d::operator=(TinyMpc2d&& other) noexcept = default;

	int TinyMpc2d::solve(const Eigen::Vector2d& x0,
	                     const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref)
	{
		if(impl_ == nullptr || impl_->solver == nullptr)
		{
			return -1;
		}

		if(!x0.allFinite())
		{
			return -1;
		}

		if(x_ref.rows() != 2 || x_ref.cols() != impl_->horizon || !x_ref.allFinite())
		{
			return -1;
		}

		// deterministic cold start：先恢复快照，再设 x0 / Xref。
		impl_->restore_cold_snapshot();

		const Eigen::VectorXd x0_dyn = x0;
		if(tiny_set_x0(impl_->solver, x0_dyn) != 0)
		{
			return -1;
		}

		const Eigen::MatrixXd x_ref_dyn = x_ref;
		tiny_set_x_ref(impl_->solver, x_ref_dyn);

		return tiny_solve(impl_->solver);
	}

	double TinyMpc2d::position(int k) const
	{
		// 原始 primal 状态（满足动力学递推），与 legacy 读 work->x 一致。
		return impl_->solver->work->x(0, k);
	}

	double TinyMpc2d::velocity(int k) const
	{
		return impl_->solver->work->x(1, k);
	}

	double TinyMpc2d::acceleration(int k) const
	{
		// 有界控制：ADMM 对输入 z（slack）施加 box [−max_acc, +max_acc]；
		// 非收敛时 primal u 可能越界，故读取 solution->u（= 投影后的 znew），
		// 保证加速度输出始终满足 |acc| <= max_acc。
		return impl_->solver->solution->u(0, k);
	}

	double TinyMpc2d::primal_acceleration(int k) const
	{
		// primal 输入 work->u（未投影）。非收敛时可能越过 max_acc。
		return impl_->solver->work->u(0, k);
	}

	bool TinyMpc2d::solved() const
	{
		return impl_->solver->solution->solved != 0;
	}

	int TinyMpc2d::iteration_count() const
	{
		return impl_->solver->solution->iter;
	}

	double TinyMpc2d::input_primal_residual() const
	{
		return impl_->solver->work->primal_residual_input;
	}

	int TinyMpc2d::horizon() const noexcept
	{
		return impl_->horizon;
	}

} // namespace tools
