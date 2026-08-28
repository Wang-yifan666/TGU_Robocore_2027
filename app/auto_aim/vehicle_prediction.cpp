/**
 * @file vehicle_prediction.cpp
 * @brief 从 TrackedTarget 快照前推整车状态并生成装甲板假设的实现。
 */

#include "app/auto_aim/vehicle_prediction.hpp"

#include <cmath>

namespace app::auto_aim
{

	namespace
	{
		constexpr double kPi = 3.14159265358979323846;
		constexpr double kTwoPi = 2.0 * kPi;
	} // namespace

	PredictedVehicle extrapolate_vehicle(const TrackedTarget& target, double dt_s)
	{
		PredictedVehicle vehicle;

		vehicle.center = target.center_in_world + target.velocity_in_world * dt_s;
		vehicle.yaw = wrap_angle(target.yaw + target.yaw_rate * dt_s);
		vehicle.radius = target.radius;
		vehicle.delta_radius = target.delta_radius;
		vehicle.delta_z = target.delta_z;
		vehicle.armor_count = static_cast<int>(target.predicted_armors.size());

		return vehicle;
	}

	PredictedVehicle predict_vehicle(const TrackedTarget& target, double dt_s)
	{
		// 公开契约：dt_s >= 0。正向前推委托给有符号底层原语，二者公式保持一致。
		return extrapolate_vehicle(target, dt_s);
	}

	std::vector<ArmorHypothesis> armor_hypotheses(const PredictedVehicle& vehicle)
	{
		std::vector<ArmorHypothesis> hypotheses;
		hypotheses.reserve(static_cast<std::size_t>(vehicle.armor_count));

		for(int i = 0; i < vehicle.armor_count; ++i)
		{
			const double theta = wrap_angle(vehicle.yaw + i * kTwoPi / vehicle.armor_count);
			const bool use_alternate = (vehicle.armor_count == 4) && (i == 1 || i == 3);
			const double radius =
			    use_alternate ? (vehicle.radius + vehicle.delta_radius) : vehicle.radius;

			ArmorHypothesis hypothesis;
			hypothesis.armor_id = i;
			hypothesis.position_in_world.x() = vehicle.center.x() - radius * std::cos(theta);
			hypothesis.position_in_world.y() = vehicle.center.y() - radius * std::sin(theta);
			hypothesis.position_in_world.z() =
			    use_alternate ? (vehicle.center.z() + vehicle.delta_z) : vehicle.center.z();
			hypothesis.yaw_in_world = theta;

			hypotheses.push_back(hypothesis);
		}

		return hypotheses;
	}

} // namespace app::auto_aim
