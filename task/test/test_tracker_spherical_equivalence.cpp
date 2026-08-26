/**
 * @file test_tracker_spherical_equivalence.cpp
 * @brief spherical measurement contract 与 legacy source（sp_vision_25 @ 604e119）的 source-equivalence fixture。
 *
 * legacy oracle 逐字复刻 legacy 公式（analytic xyz2ypd_jacobian、while-loop limit_rad、
 * h_armor_xyz / h_jacobian chain rule、adaptive R_dig），不调用任何 production helper，
 * 从而避免 self-validation（production helper 不自证 golden）。
 */

#include "app/auto_aim/target.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include <Eigen/Dense>
#include "test_logging.hpp"

namespace
{

	constexpr double kPi = 3.14159265358979323846;
	constexpr double kTwoPi = 2.0 * kPi;

	// ============================================================
	// legacy oracle（sp_vision_25 pinned commit 604e11943b7dc3a8b466cce6d72079e5d271b2b7）
	// ============================================================

	double legacy_limit_rad(double angle)
	{
		while(angle > kPi)
		{
			angle -= 2.0 * kPi;
		}
		while(angle <= -kPi)
		{
			angle += 2.0 * kPi;
		}
		return angle;
	}

	Eigen::Vector3d legacy_xyz2ypd(const Eigen::Vector3d& xyz)
	{
		const double x = xyz[0];
		const double y = xyz[1];
		const double z = xyz[2];
		const double yaw = std::atan2(y, x);
		const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
		const double distance = std::sqrt(x * x + y * y + z * z);
		return {yaw, pitch, distance};
	}

	Eigen::MatrixXd legacy_xyz2ypd_jacobian(const Eigen::Vector3d& xyz)
	{
		const double x = xyz[0];
		const double y = xyz[1];
		const double z = xyz[2];

		const double dyaw_dx = -y / (x * x + y * y);
		const double dyaw_dy = x / (x * x + y * y);
		const double dyaw_dz = 0.0;

		const double dpitch_dx =
		    -(x * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
		const double dpitch_dy =
		    -(y * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
		const double dpitch_dz =
		    1 / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 0.5));

		const double ddistance_dx = x / std::pow((x * x + y * y + z * z), 0.5);
		const double ddistance_dy = y / std::pow((x * x + y * y + z * z), 0.5);
		const double ddistance_dz = z / std::pow((x * x + y * y + z * z), 0.5);

		Eigen::MatrixXd J(3, 3);
		// clang-format off
J << dyaw_dx, dyaw_dy, dyaw_dz,
dpitch_dx, dpitch_dy, dpitch_dz,
ddistance_dx, ddistance_dy, ddistance_dz;
		// clang-format on
		return J;
	}

	Eigen::Vector3d legacy_h_armor_xyz(const Eigen::VectorXd& x, int id, int armor_num)
	{
		const double angle = legacy_limit_rad(x[6] + id * 2.0 * kPi / armor_num);
		const bool use_l_h = (armor_num == 4) && (id == 1 || id == 3);
		const double r = use_l_h ? (x[8] + x[9]) : x[8];
		const double armor_x = x[0] - r * std::cos(angle);
		const double armor_y = x[2] - r * std::sin(angle);
		const double armor_z = use_l_h ? (x[4] + x[10]) : x[4];
		return {armor_x, armor_y, armor_z};
	}

	Eigen::MatrixXd legacy_h_jacobian(const Eigen::VectorXd& x, int id, int armor_num)
	{
		const double angle = legacy_limit_rad(x[6] + id * 2.0 * kPi / armor_num);
		const bool use_l_h = (armor_num == 4) && (id == 1 || id == 3);
		const double r = use_l_h ? (x[8] + x[9]) : x[8];

		const double dx_da = r * std::sin(angle);
		const double dy_da = -r * std::cos(angle);
		const double dx_dr = -std::cos(angle);
		const double dy_dr = -std::sin(angle);
		const double dx_dl = use_l_h ? -std::cos(angle) : 0.0;
		const double dy_dl = use_l_h ? -std::sin(angle) : 0.0;
		const double dz_dh = use_l_h ? 1.0 : 0.0;

		Eigen::MatrixXd H_armor_xyza(4, 11);
		H_armor_xyza.setZero();
		H_armor_xyza(0, 0) = 1.0;
		H_armor_xyza(0, 6) = dx_da;
		H_armor_xyza(0, 8) = dx_dr;
		H_armor_xyza(0, 9) = dx_dl;
		H_armor_xyza(1, 2) = 1.0;
		H_armor_xyza(1, 6) = dy_da;
		H_armor_xyza(1, 8) = dy_dr;
		H_armor_xyza(1, 9) = dy_dl;
		H_armor_xyza(2, 4) = 1.0;
		H_armor_xyza(2, 10) = dz_dh;
		H_armor_xyza(3, 6) = 1.0;

		const Eigen::Vector3d armor_xyz = legacy_h_armor_xyz(x, id, armor_num);
		const Eigen::MatrixXd H_armor_ypd = legacy_xyz2ypd_jacobian(armor_xyz);

		Eigen::MatrixXd H_armor_ypda(4, 4);
		H_armor_ypda.setZero();
		H_armor_ypda.topLeftCorner(3, 3) = H_armor_ypd;
		H_armor_ypda(3, 3) = 1.0;

		return H_armor_ypda * H_armor_xyza;
	}

	Eigen::VectorXd legacy_R_dig(const app::auto_aim::ArmorObservation& observation)
	{
		const double center_yaw =
		    std::atan2(observation.position_in_world.y(), observation.position_in_world.x());
		const double delta_angle = legacy_limit_rad(observation.armor_yaw_in_world - center_yaw);
		const double distance = observation.ypd_in_world.z();

		Eigen::VectorXd R_dig(4);
		R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
		    std::log(std::abs(distance) + 1.0) / 200.0 + 9e-2;
		return R_dig;
	}

	// ============================================================
	// test runner
	// ============================================================

	class TestRunner
	{
	public:
		void begin(const std::string& name)
		{
			current_test_ = name;
			current_test_failed_ = false;
			std::printf("===== %s =====\n", name.c_str());
		}

		void expect(bool condition, const std::string& message)
		{
			++check_count_;
			if(condition)
			{
				std::printf("[PASS] %s\n", message.c_str());
				return;
			}
			++failure_count_;
			current_test_failed_ = true;
			std::printf("[FAIL] %s\n", message.c_str());
		}

		void end()
		{
			std::printf("[%s] %s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
			            current_test_.c_str());
		}

		[[nodiscard]] int failure_count() const noexcept
		{
			return failure_count_;
		}

		void print_summary() const
		{
			std::printf("========================================\n");
			std::printf("Checks:   %d\n", check_count_);
			std::printf("Failures: %d\n", failure_count_);
			std::printf("========================================\n");
		}

	private:
		std::string current_test_;
		int check_count_ = 0;
		int failure_count_ = 0;
		bool current_test_failed_ = false;
	};

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	bool mat_near(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs, double eps)
	{
		return (lhs - rhs).norm() <= eps;
	}

	// ============================================================
	// fixtures
	// ============================================================

	app::auto_aim::ArmorObservation make_obs(double x, double y, double z, double armor_yaw,
	                                         app::auto_aim::ArmorColor color,
	                                         app::auto_aim::ArmorName name,
	                                         app::auto_aim::ArmorType type)
	{
		app::auto_aim::ArmorObservation o;
		o.color = color;
		o.name = name;
		o.type = type;
		o.position_in_world = Eigen::Vector3d(x, y, z);
		o.armor_yaw_in_world = armor_yaw;
		o.ypd_in_world = legacy_xyz2ypd(o.position_in_world);
		return o;
	}

	Eigen::VectorXd make_state(double cx, double cy, double cz, double yaw, double radius,
	                           double delta_radius, double delta_z)
	{
		Eigen::VectorXd x = Eigen::VectorXd::Zero(11);
		x(0) = cx;
		x(1) = 0.1;
		x(2) = cy;
		x(3) = 0.2;
		x(4) = cz;
		x(5) = 0.3;
		x(6) = yaw;
		x(7) = 0.4;
		x(8) = radius;
		x(9) = delta_radius;
		x(10) = delta_z;
		return x;
	}

	app::auto_aim::TargetModelConfig model_config()
	{
		app::auto_aim::TargetModelConfig c;
		c.translation_accel_variance = 1.0;
		c.yaw_accel_variance = 1.0;
		c.radius_random_walk_variance = 1.0;
		c.delta_radius_random_walk_variance = 1.0;
		c.delta_z_random_walk_variance = 1.0;
		return c;
	}

	app::auto_aim::Target make_target(const app::auto_aim::ArmorObservation& o, double radius)
	{
		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11);
		return app::auto_aim::Target(o, radius, P0, model_config());
	}

	app::auto_aim::MeasurementNoiseConfig legacy_default_noise()
	{
		app::auto_aim::MeasurementNoiseConfig config;
		config.base_covariance = Eigen::MatrixXd::Zero(4, 4);
		config.base_covariance(0, 0) = 4e-3;
		config.base_covariance(1, 1) = 4e-3;
		config.base_covariance(2, 2) = 1.0;
		config.base_covariance(3, 3) = 9e-2;
		config.distance_angle_log_gain = 1.0;
		config.armor_yaw_distance_log_gain = 1.0 / 200.0;
		return config;
	}

	// ============================================================
	// tests
	// ============================================================

	void test_equivalence_4armor(TestRunner& runner)
	{
		runner.begin("Equivalence 4-armor");

		{
			auto o = make_obs(0.8, 1.9, 0.5, 0.3, app::auto_aim::ArmorColor::Red,
			                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
			app::auto_aim::Target t = make_target(o, 0.2);
			Eigen::VectorXd x = make_state(1.0, 2.0, 0.5, 0.3, 0.2, 0.0, 0.0);

			const Eigen::Vector4d h = t.measurement_model(x, 0);
			const Eigen::Vector3d armor_xyz = legacy_h_armor_xyz(x, 0, 4);
			const Eigen::Vector3d ypd = legacy_xyz2ypd(armor_xyz);
			runner.expect(near(h.x(), ypd.x()), "normal armor0 bearing_yaw");
			runner.expect(near(h.y(), ypd.y()), "normal armor0 pitch");
			runner.expect(near(h.z(), ypd.z()), "normal armor0 distance");
			runner.expect(near(h.w(), legacy_limit_rad(x(6)), 1e-9), "normal armor0 armor_yaw");
		}

		{
			auto o = make_obs(0.8, 1.9, 0.6, 0.3, app::auto_aim::ArmorColor::Red,
			                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
			app::auto_aim::Target t = make_target(o, 0.2);
			Eigen::VectorXd x = make_state(1.0, 2.0, 0.5, 0.3, 0.2, 0.05, 0.1);

			const Eigen::Vector4d h = t.measurement_model(x, 1);
			const Eigen::Vector3d armor_xyz = legacy_h_armor_xyz(x, 1, 4);
			const Eigen::Vector3d ypd = legacy_xyz2ypd(armor_xyz);
			runner.expect(near(h.x(), ypd.x()), "alternate armor1 bearing_yaw");
			runner.expect(near(h.y(), ypd.y()), "alternate armor1 pitch");
			runner.expect(near(h.z(), ypd.z()), "alternate armor1 distance");
			runner.expect(near(h.w(), legacy_limit_rad(x(6) + 2.0 * kPi / 4.0), 1e-9),
			              "alternate armor1 armor_yaw");
		}

		runner.end();
	}

	void test_equivalence_armor_counts(TestRunner& runner)
	{
		runner.begin("Equivalence 3/2-armor");

		{
			auto o = make_obs(1.0, 1.0, 0.0, 0.4, app::auto_aim::ArmorColor::Red,
			                  app::auto_aim::ArmorName::Outpost, app::auto_aim::ArmorType::Big);
			app::auto_aim::Target t = make_target(o, 0.2765);
			Eigen::VectorXd x = make_state(1.0, 1.0, 0.0, 0.4, 0.2765, 0.0, 0.0);

			const Eigen::Vector4d h = t.measurement_model(x, 1);
			const Eigen::Vector3d armor_xyz = legacy_h_armor_xyz(x, 1, 3);
			const Eigen::Vector3d ypd = legacy_xyz2ypd(armor_xyz);
			runner.expect(near(h.x(), ypd.x()), "3-armor bearing_yaw");
			runner.expect(near(h.y(), ypd.y()), "3-armor pitch");
			runner.expect(near(h.z(), ypd.z()), "3-armor distance");
		}

		{
			auto o = make_obs(1.0, 1.0, 0.0, 0.4, app::auto_aim::ArmorColor::Blue,
			                  app::auto_aim::ArmorName::Three, app::auto_aim::ArmorType::Big);
			app::auto_aim::Target t = make_target(o, 0.2);
			Eigen::VectorXd x = make_state(1.0, 1.0, 0.0, 0.4, 0.2, 0.0, 0.0);

			const Eigen::Vector4d h = t.measurement_model(x, 1);
			const Eigen::Vector3d armor_xyz = legacy_h_armor_xyz(x, 1, 2);
			const Eigen::Vector3d ypd = legacy_xyz2ypd(armor_xyz);
			runner.expect(near(h.x(), ypd.x()), "2-armor bearing_yaw");
			runner.expect(near(h.y(), ypd.y()), "2-armor pitch");
			runner.expect(near(h.z(), ypd.z()), "2-armor distance");
		}

		runner.end();
	}

	void test_equivalence_jacobian(TestRunner& runner)
	{
		runner.begin("Equivalence analytic H");

		auto o = make_obs(0.8, 1.9, 0.6, 0.3, app::auto_aim::ArmorColor::Red,
		                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
		app::auto_aim::Target t = make_target(o, 0.2);
		Eigen::VectorXd x = make_state(1.0, 2.0, 0.5, 0.3, 0.2, 0.05, 0.1);

		const Eigen::MatrixXd H = t.measurement_jacobian(x, 1);
		const Eigen::MatrixXd H_legacy = legacy_h_jacobian(x, 1, 4);
		runner.expect(mat_near(H, H_legacy, 1e-9), "H == legacy chain rule (alternate)");

		runner.end();
	}

	void test_equivalence_residual(TestRunner& runner)
	{
		runner.begin("Equivalence residual");

		// 覆盖精确 ±pi 边界（判别 limit_rad vs wrap_angle）与普通大角度。
		const double z_cases[][4] = {
		    {kPi, 0.0, 1.0, 0.0},   // bearing_yaw raw +pi
		    {-kPi, 0.0, 1.0, 0.0},  // bearing_yaw raw -pi
		    {0.0, kPi, 1.0, 0.0},   // pitch raw +pi
		    {0.0, -kPi, 1.0, 0.0},  // pitch raw -pi
		    {0.0, 0.0, 1.0, kPi},   // armor_yaw raw +pi
		    {0.0, 0.0, 1.0, -kPi},  // armor_yaw raw -pi
		    {3.0, -0.2, 5.0, 3.0}}; // 普通大角度

		const double h_cases[][4] = {
		    {0.0, 0.0, 1.0, 0.0},
		    {0.0, 0.0, 1.0, 0.0},
		    {0.0, 0.0, 1.0, 0.0},
		    {0.0, 0.0, 1.0, 0.0},
		    {0.0, 0.0, 1.0, 0.0},
		    {0.0, 0.0, 1.0, 0.0},
		    {-3.0, 0.1, 3.0, -3.0}};

		constexpr int kCaseCount = 7;

		for(int i = 0; i < kCaseCount; ++i)
		{
			Eigen::VectorXd z(4);
			Eigen::VectorXd h(4);
			z << z_cases[i][0], z_cases[i][1], z_cases[i][2], z_cases[i][3];
			h << h_cases[i][0], h_cases[i][1], h_cases[i][2], h_cases[i][3];

			const Eigen::VectorXd r = app::auto_aim::Target::measurement_residual(z, h);

			Eigen::VectorXd expected = z - h;
			expected(0) = legacy_limit_rad(expected(0));
			expected(1) = legacy_limit_rad(expected(1));
			expected(3) = legacy_limit_rad(expected(3));

			runner.expect((r - expected).norm() <= 1e-12, "residual == legacy z_subtract");
		}

		runner.end();
	}

	void test_equivalence_R(TestRunner& runner)
	{
		runner.begin("Equivalence adaptive R");

		const double cases[][4] = {
		    {1.0, 1.0, 0.0, 0.5}, {3.0, 0.2, 0.3, -2.0}, {0.5, 0.5, 0.5, 1.2}};

		for(const auto& c: cases)
		{
			auto o = make_obs(c[0], c[1], c[2], c[3], app::auto_aim::ArmorColor::Red,
			                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);

			const Eigen::MatrixXd R =
			    app::auto_aim::Target::measurement_covariance(o, legacy_default_noise());
			const Eigen::VectorXd R_dig = legacy_R_dig(o);

			runner.expect(near(R(0, 0), 4e-3, 1e-12), "R00 constant 4e-3");
			runner.expect(near(R(1, 1), 4e-3, 1e-12), "R11 constant 4e-3");
			runner.expect(near(R(2, 2), R_dig(2), 1e-12), "R22 == legacy log(delta_angle)");
			runner.expect(near(R(3, 3), R_dig(3), 1e-12), "R33 == legacy log(distance)");
		}

		runner.end();
	}

	void test_equivalence_z(TestRunner& runner)
	{
		runner.begin("Equivalence z (measurement_vector)");

		auto o = make_obs(1.5, -0.8, 0.4, -0.7, app::auto_aim::ArmorColor::Red,
		                  app::auto_aim::ArmorName::Four, app::auto_aim::ArmorType::Small);
		const Eigen::Vector4d z = app::auto_aim::Target::measurement_vector(o);

		runner.expect(near(z.x(), o.ypd_in_world.x(), 1e-12), "z[0] == ypd[0]");
		runner.expect(near(z.y(), o.ypd_in_world.y(), 1e-12), "z[1] == ypd[1]");
		runner.expect(near(z.z(), o.ypd_in_world.z(), 1e-12), "z[2] == ypd[2]");
		runner.expect(near(z.w(), o.armor_yaw_in_world, 1e-12), "z[3] == armor_yaw");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_tracker_spherical_equivalence");
	std::printf("=== Tracker Spherical Source-Equivalence Test Suite ===\n\n");

	TestRunner runner;

	test_equivalence_z(runner);
	test_equivalence_4armor(runner);
	test_equivalence_armor_counts(runner);
	test_equivalence_jacobian(runner);
	test_equivalence_residual(runner);
	test_equivalence_R(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Tracker spherical equivalence tests failed ===\n");
		return 1;
	}

	std::printf("=== All tracker spherical equivalence tests passed ===\n");
	return 0;
}
