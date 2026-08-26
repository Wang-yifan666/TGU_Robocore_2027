/**
 * @file target.cpp
 * @brief 敌方车辆的 Target 数学模型（11D 车辆中心状态）实现。
 */

#include "app/auto_aim/target.hpp"

#include <cmath>
#include <stdexcept>

#include "tools/maths_tools.hpp"

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

		// measurement residual：index 0/1/3（bearing_yaw / pitch / armor_yaw）wrap，index 2 不 wrap。
		// 直接复用 production helper，避免在构造器里复制公式。
		auto residual = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
			return Target::measurement_residual(a, b);
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

	Eigen::Vector3d Target::armor_position(const Eigen::VectorXd& x, int armor_id) const
	{
		const ArmorGeometry g = geometry(x, armor_id);

		Eigen::Vector3d position;
		position.x() = x(kStateX) - g.radius * std::cos(g.theta);
		position.y() = x(kStateY) - g.radius * std::sin(g.theta);
		position.z() = g.use_alternate ? (x(kStateZ) + x(kStateDeltaZ)) : x(kStateZ);

		return position;
	}

	Eigen::Vector4d Target::measurement_model(const Eigen::VectorXd& x, int armor_id) const
	{
		if(x.size() != kTargetStateDim || x.cols() != 1 || !x.allFinite())
		{
			throw std::invalid_argument("x must be (11 x 1) and finite");
		}

		if(armor_id < 0 || armor_id >= armor_count_)
		{
			throw std::invalid_argument("armor_id out of range");
		}

		const ArmorGeometry g = geometry(x, armor_id);
		const Eigen::Vector3d position = armor_position(x, armor_id);
		const Eigen::Vector3d ypd = tools::maths_tools::xyz2ypd(position);

		return Eigen::Vector4d(ypd.x(), ypd.y(), ypd.z(), tools::maths_tools::limit_rad(g.theta));
	}

	Eigen::Vector4d Target::measurement_vector(const ArmorObservation& observation)
	{
		if(!observation.ypd_in_world.allFinite())
		{
			throw std::invalid_argument("observation ypd_in_world must be finite");
		}

		if(!std::isfinite(observation.armor_yaw_in_world))
		{
			throw std::invalid_argument("observation armor_yaw_in_world must be finite");
		}

		return Eigen::Vector4d(observation.ypd_in_world.x(), observation.ypd_in_world.y(),
		                       observation.ypd_in_world.z(), observation.armor_yaw_in_world);
	}

	Eigen::VectorXd Target::measurement_residual(const Eigen::VectorXd& z, const Eigen::VectorXd& h)
	{
		if(z.size() != kTargetMeasurementDim || z.cols() != 1 || !z.allFinite())
		{
			throw std::invalid_argument("z must be (4 x 1) and finite");
		}

		if(h.size() != kTargetMeasurementDim || h.cols() != 1 || !h.allFinite())
		{
			throw std::invalid_argument("h must be (4 x 1) and finite");
		}

		Eigen::VectorXd c = z - h;
		c(0) = tools::maths_tools::limit_rad(c(0));
		c(1) = tools::maths_tools::limit_rad(c(1));
		c(3) = tools::maths_tools::limit_rad(c(3));

		return c;
	}

	Eigen::MatrixXd Target::measurement_covariance(const ArmorObservation& observation,
	                                               const MeasurementNoiseConfig& config)
	{
		if(config.base_covariance.rows() != kTargetMeasurementDim
		   || config.base_covariance.cols() != kTargetMeasurementDim)
		{
			throw std::invalid_argument("base_covariance must be 4x4");
		}

		if(!observation.position_in_world.allFinite())
		{
			throw std::invalid_argument("observation position_in_world must be finite");
		}

		if(!observation.ypd_in_world.allFinite())
		{
			throw std::invalid_argument("observation ypd_in_world must be finite");
		}

		if(!std::isfinite(observation.armor_yaw_in_world))
		{
			throw std::invalid_argument("observation armor_yaw_in_world must be finite");
		}

		if(!std::isfinite(config.distance_angle_log_gain) || config.distance_angle_log_gain < 0.0)
		{
			throw std::invalid_argument("distance_angle_log_gain must be finite and >= 0");
		}

		if(!std::isfinite(config.armor_yaw_distance_log_gain)
		   || config.armor_yaw_distance_log_gain < 0.0)
		{
			throw std::invalid_argument("armor_yaw_distance_log_gain must be finite and >= 0");
		}

		Eigen::MatrixXd R = config.base_covariance;

		// delta_angle 用 observation（armor_yaw vs 观测 bearing），不依赖 predicted state。
		const double center_yaw =
		    std::atan2(observation.position_in_world.y(), observation.position_in_world.x());
		const double delta_angle =
		    tools::maths_tools::limit_rad(observation.armor_yaw_in_world - center_yaw);

		// distance 为 spherical distance（单位 m），log1p 内逐字保留米数值（legacy 语义）。
		const double distance = observation.ypd_in_world.z();

		// adaptive 项只作用于对角：R(2,2) 是 distance variance（m^2），R(3,3) 是 armor_yaw variance（rad^2）。
		R(2, 2) += config.distance_angle_log_gain * std::log1p(std::abs(delta_angle));
		R(3, 3) += config.armor_yaw_distance_log_gain * std::log1p(std::abs(distance));

		return R;
	}

	bool Target::correct(const ArmorObservation& observation, int armor_id,
	                     const MeasurementNoiseConfig& measurement_noise)
	{
		if(armor_id < 0 || armor_id >= armor_count_)
		{
			throw std::invalid_argument("armor_id out of range");
		}

		// has_armor_switch：本生命周期内是否匹配到过非 0 装甲板（对应 SP25 jumped）。
		// 置位发生在 EKF update 之前，与 SP25 在 update() 内按匹配 id 置位一致。
		if(armor_id != 0)
		{
			has_armor_switch_ = true;
		}

		// z / h / H / R 全部来自直接可测的 production helper，不在 correct 内重复公式。
		const Eigen::VectorXd z = measurement_vector(observation);

		const Eigen::VectorXd& x_prior = ekf_->state();
		const Eigen::MatrixXd H = measurement_jacobian(x_prior, armor_id);

		auto h = [this, armor_id](const Eigen::VectorXd& x) {
			return this->measurement_model(x, armor_id);
		};

		const Eigen::MatrixXd R = measurement_covariance(observation, measurement_noise);

		return ekf_->update(z, H, R, h);
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
		if(x.size() != kTargetStateDim || x.cols() != 1 || !x.allFinite())
		{
			throw std::invalid_argument("x must be (11 x 1) and finite");
		}

		if(armor_id < 0 || armor_id >= armor_count_)
		{
			throw std::invalid_argument("armor_id out of range");
		}

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

		// J_xyza_state：d[armor_x, armor_y, armor_z, armor_yaw] / d(state)（4x11）。
		Eigen::MatrixXd J_xyza_state = Eigen::MatrixXd::Zero(kTargetMeasurementDim, kTargetStateDim);

		J_xyza_state(0, kStateX) = 1.0;
		J_xyza_state(0, kStateYaw) = dx_dyaw;
		J_xyza_state(0, kStateRadius) = dx_dr;
		J_xyza_state(0, kStateDeltaRadius) = dx_dl;

		J_xyza_state(1, kStateY) = 1.0;
		J_xyza_state(1, kStateYaw) = dy_dyaw;
		J_xyza_state(1, kStateRadius) = dy_dr;
		J_xyza_state(1, kStateDeltaRadius) = dy_dl;

		J_xyza_state(2, kStateZ) = 1.0;
		J_xyza_state(2, kStateDeltaZ) = dz_dh;

		J_xyza_state(3, kStateYaw) = 1.0;

		// J_ypda_xyza：d[ypd, armor_yaw] / d[armor_x, armor_y, armor_z, armor_yaw]（4x4）。
		const Eigen::Vector3d position = armor_position(x, armor_id);
		const Eigen::MatrixXd J_ypd = tools::maths_tools::xyz2ypd_jacobian(position);

		Eigen::MatrixXd J_ypda_xyza =
		    Eigen::MatrixXd::Zero(kTargetMeasurementDim, kTargetMeasurementDim);
		J_ypda_xyza.topLeftCorner(3, 3) = J_ypd;
		J_ypda_xyza(3, 3) = 1.0;

		return J_ypda_xyza * J_xyza_state;
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

	bool Target::has_armor_switch() const noexcept
	{
		return has_armor_switch_;
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