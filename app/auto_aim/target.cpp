/**
 * @file target.cpp
 * @brief 敌方车辆的 Target 数学模型（11D 车辆中心状态）实现。
 */

#include "app/auto_aim/target.hpp"

#include <cmath>
#include <stdexcept>

namespace app::auto_aim
{

	namespace
	{
		constexpr double kPi = 3.14159265358979323846;
		constexpr double kTwoPi = 2.0 * kPi;
	} // namespace

	int armor_count_for(ArmorType type, ArmorName name)
	{
		if(type == ArmorType::Big
		   && (name == ArmorName::Three || name == ArmorName::Four || name == ArmorName::Five))
		{
			return 2;
		}

		if(name == ArmorName::Outpost)
		{
			return 3;
		}

		if(name == ArmorName::Base)
		{
			return 3;
		}

		return 4;
	}

	double wrap_angle(double angle)
	{
		double wrapped = std::fmod(angle + kPi, kTwoPi);

		if(wrapped < 0.0)
		{
			wrapped += kTwoPi;
		}

		return wrapped - kPi;
	}

	Eigen::MatrixXd Target::transition_matrix(double dt)
	{
		Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kTargetStateDim, kTargetStateDim);

		F(kStateX, kStateVx) = dt;
		F(kStateY, kStateVy) = dt;
		F(kStateZ, kStateVz) = dt;
		F(kStateYaw, kStateYawRate) = dt;

		return F;
	}

	Eigen::MatrixXd Target::process_noise_matrix(double dt, const TargetModelConfig& config)
	{
		Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kTargetStateDim, kTargetStateDim);

		const double dt2 = dt * dt;
		const double dt3 = dt2 * dt;
		const double dt4 = dt2 * dt2;

		// 平动加速度离散白噪声 block：[dt^4/4, dt^3/2; dt^3/2, dt^2] * variance。
		const double q_pos = config.translation_accel_variance * dt4 / 4.0;
		const double q_cross = config.translation_accel_variance * dt3 / 2.0;
		const double q_vel = config.translation_accel_variance * dt2;

		Q(kStateX, kStateX) = q_pos;
		Q(kStateX, kStateVx) = q_cross;
		Q(kStateVx, kStateX) = q_cross;
		Q(kStateVx, kStateVx) = q_vel;

		Q(kStateY, kStateY) = q_pos;
		Q(kStateY, kStateVy) = q_cross;
		Q(kStateVy, kStateY) = q_cross;
		Q(kStateVy, kStateVy) = q_vel;

		Q(kStateZ, kStateZ) = q_pos;
		Q(kStateZ, kStateVz) = q_cross;
		Q(kStateVz, kStateZ) = q_cross;
		Q(kStateVz, kStateVz) = q_vel;

		// 偏航角加速度离散白噪声 block。
		const double q_yaw_pos = config.yaw_accel_variance * dt4 / 4.0;
		const double q_yaw_cross = config.yaw_accel_variance * dt3 / 2.0;
		const double q_yaw_rate = config.yaw_accel_variance * dt2;

		Q(kStateYaw, kStateYaw) = q_yaw_pos;
		Q(kStateYaw, kStateYawRate) = q_yaw_cross;
		Q(kStateYawRate, kStateYaw) = q_yaw_cross;
		Q(kStateYawRate, kStateYawRate) = q_yaw_rate;

		// geometry 状态用 variance * dt 作为 random walk。
		Q(kStateRadius, kStateRadius) = config.radius_random_walk_variance * dt;
		Q(kStateDeltaRadius, kStateDeltaRadius) = config.delta_radius_random_walk_variance * dt;
		Q(kStateDeltaZ, kStateDeltaZ) = config.delta_z_random_walk_variance * dt;

		return Q;
	}

	Target::Target(const ArmorObservation& observation, double initial_radius,
	               const Eigen::MatrixXd& initial_covariance, const TargetModelConfig& config):
	config_(config),
	color_(observation.color),
	name_(observation.name),
	type_(observation.type),
	priority_(observation.priority),
	armor_count_(armor_count_for(observation.type, observation.name))
	{
		if(!observation.position_in_world.allFinite())
		{
			throw std::invalid_argument("observation position_in_world must be finite");
		}

		if(!std::isfinite(observation.armor_yaw_in_world))
		{
			throw std::invalid_argument("observation armor_yaw_in_world must be finite");
		}

		if(!std::isfinite(initial_radius) || initial_radius <= 0.0)
		{
			throw std::invalid_argument("initial_radius must be finite and > 0");
		}

		if(initial_covariance.rows() != kTargetStateDim
		   || initial_covariance.cols() != kTargetStateDim)
		{
			throw std::invalid_argument("initial_covariance must be 11x11");
		}

		auto valid_variance = [](double v) {
			return std::isfinite(v) && v >= 0.0;
		};

		if(!valid_variance(config.translation_accel_variance)
		   || !valid_variance(config.yaw_accel_variance)
		   || !valid_variance(config.radius_random_walk_variance)
		   || !valid_variance(config.delta_radius_random_walk_variance)
		   || !valid_variance(config.delta_z_random_walk_variance))
		{
			throw std::invalid_argument(
			    "process noise variances must be finite and non-negative");
		}

		const double observed_yaw = wrap_angle(observation.armor_yaw_in_world);

		const double center_x =
		    observation.position_in_world.x() + initial_radius * std::cos(observed_yaw);
		const double center_y =
		    observation.position_in_world.y() + initial_radius * std::sin(observed_yaw);

		Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kTargetStateDim);

		x0(kStateX) = center_x;
		x0(kStateY) = center_y;
		x0(kStateZ) = observation.position_in_world.z();
		x0(kStateYaw) = observed_yaw;
		x0(kStateRadius) = initial_radius;

		// update 时 yaw 分量归一化。
		auto state_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			Eigen::VectorXd c = a + b;
			c(kStateYaw) = wrap_angle(c(kStateYaw));
			return c;
		};

		// measurement residual：只有 yaw（index 3）wrap，position 不 wrap。
		auto residual = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			Eigen::VectorXd c = a - b;
			c(3) = wrap_angle(c(3));
			return c;
		};

		ekf_.emplace(x0, initial_covariance, state_add, residual);
	}

	void Target::predict(double dt)
	{
		if(!std::isfinite(dt) || dt < 0.0)
		{
			throw std::invalid_argument("dt must be finite and >= 0");
		}

		const Eigen::MatrixXd F = transition_matrix(dt);
		const Eigen::MatrixXd Q = process_noise_matrix(dt, config_);

		// yaw 预测后归一化到 [-pi, pi)。
		auto f = [&F](const Eigen::VectorXd& x) {
			Eigen::VectorXd x_next = F * x;
			x_next(kStateYaw) = wrap_angle(x_next(kStateYaw));
			return x_next;
		};

		ekf_->predict(F, Q, f);
	}

	std::vector<ArmorHypothesis> Target::armor_hypotheses() const
	{
		const Eigen::VectorXd& x = ekf_->state();

		std::vector<ArmorHypothesis> hypotheses;
		hypotheses.reserve(static_cast<std::size_t>(armor_count_));

		for(int i = 0; i < armor_count_; ++i)
		{
			const ArmorGeometry g = geometry(x, i);

			ArmorHypothesis h;
			h.armor_id = i;
			h.position_in_world.x() = x(kStateX) - g.radius * std::cos(g.theta);
			h.position_in_world.y() = x(kStateY) - g.radius * std::sin(g.theta);
			h.position_in_world.z() = g.use_alternate
			                        ? (x(kStateZ) + x(kStateDeltaZ))
			                        : x(kStateZ);
			h.yaw_in_world = g.theta;
			hypotheses.push_back(h);
		}

		return hypotheses;
	}

	Eigen::Vector4d Target::measurement_model(const Eigen::VectorXd& x, int armor_id) const
	{
		const ArmorGeometry g = geometry(x, armor_id);

		Eigen::Vector4d z;
		z.x() = x(kStateX) - g.radius * std::cos(g.theta);
		z.y() = x(kStateY) - g.radius * std::sin(g.theta);
		z.z() = g.use_alternate ? (x(kStateZ) + x(kStateDeltaZ)) : x(kStateZ);
		z.w() = g.theta;

		return z;
	}

	bool Target::correct(const ArmorObservation& observation, int armor_id,
	                     const Eigen::MatrixXd& measurement_covariance)
	{
		if(armor_id < 0 || armor_id >= armor_count_)
		{
			throw std::invalid_argument("armor_id out of range");
		}

		if(!observation.position_in_world.allFinite())
		{
			throw std::invalid_argument("observation position_in_world must be finite");
		}

		if(!std::isfinite(observation.armor_yaw_in_world))
		{
			throw std::invalid_argument("observation armor_yaw_in_world must be finite");
		}

		if(measurement_covariance.rows() != kTargetMeasurementDim
		   || measurement_covariance.cols() != kTargetMeasurementDim)
		{
			throw std::invalid_argument("measurement_covariance must be 4x4");
		}

		const Eigen::VectorXd& x_prior = ekf_->state();

		const Eigen::MatrixXd H = measurement_jacobian(x_prior, armor_id);

		auto h = [this, armor_id](const Eigen::VectorXd& x) {
			return this->measurement_model(x, armor_id);
		};

		Eigen::VectorXd z(kTargetMeasurementDim);
		z(0) = observation.position_in_world.x();
		z(1) = observation.position_in_world.y();
		z(2) = observation.position_in_world.z();
		z(3) = observation.armor_yaw_in_world;

		return ekf_->update(z, H, measurement_covariance, h);
	}

	const Eigen::VectorXd& Target::last_innovation() const noexcept
	{
		return ekf_->last_innovation();
	}

	double Target::last_nis() const noexcept
	{
		return ekf_->last_nis();
	}

	Eigen::MatrixXd Target::measurement_jacobian(const Eigen::VectorXd& x, int armor_id) const
	{
		const ArmorGeometry g = geometry(x, armor_id);

		const double sin_theta = std::sin(g.theta);
		const double cos_theta = std::cos(g.theta);

		const double dx_dyaw = g.radius * sin_theta;
		const double dy_dyaw = -g.radius * cos_theta;

		const double dx_dr = -cos_theta;
		const double dy_dr = -sin_theta;

		const double dx_dl = g.use_alternate ? -cos_theta : 0.0;
		const double dy_dl = g.use_alternate ? -sin_theta : 0.0;
		const double dz_dh = g.use_alternate ? 1.0 : 0.0;

		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kTargetMeasurementDim, kTargetStateDim);

		H(0, kStateX) = 1.0;
		H(0, kStateYaw) = dx_dyaw;
		H(0, kStateRadius) = dx_dr;
		H(0, kStateDeltaRadius) = dx_dl;

		H(1, kStateY) = 1.0;
		H(1, kStateYaw) = dy_dyaw;
		H(1, kStateRadius) = dy_dr;
		H(1, kStateDeltaRadius) = dy_dl;

		H(2, kStateZ) = 1.0;
		H(2, kStateDeltaZ) = dz_dh;

		H(3, kStateYaw) = 1.0;

		return H;
	}

	const Eigen::VectorXd& Target::state() const noexcept
	{
		return ekf_->state();
	}

	const Eigen::MatrixXd& Target::covariance() const noexcept
	{
		return ekf_->covariance();
	}

	int Target::armor_count() const noexcept
	{
		return armor_count_;
	}

	ArmorColor Target::color() const noexcept
	{
		return color_;
	}

	ArmorName Target::name() const noexcept
	{
		return name_;
	}

	ArmorType Target::type() const noexcept
	{
		return type_;
	}

	ArmorPriority Target::priority() const noexcept
	{
		return priority_;
	}

	Target::ArmorGeometry Target::geometry(const Eigen::VectorXd& x, int armor_id) const
	{
		ArmorGeometry g;
		g.theta = wrap_angle(x(kStateYaw) + armor_id * kTwoPi / armor_count_);
		g.use_alternate = (armor_count_ == 4) && (armor_id == 1 || armor_id == 3);
		g.radius = g.use_alternate ? (x(kStateRadius) + x(kStateDeltaRadius)) : x(kStateRadius);
		return g;
	}

} // namespace app::auto_aim