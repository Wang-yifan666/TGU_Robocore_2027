/**
 * @file test_vehicle_prediction.cpp
 * @brief vehicle_prediction 纯几何前推 helper 单元测试。
 *
 * 覆盖：
 * - dt == 0 时与 Target::armor_hypotheses() 等价（防止几何漂移）；
 * - dt > 0 平移前推；
 * - dt > 0 旋转前推（含 yaw 归一化）；
 * - 4 装甲板交替 radius / z；
 * - armor indexing（armor_count == 2 / 3 / 4）；
 * - armor_count 取自 predicted_armors.size()。
 */

#include "app/auto_aim/vehicle_prediction.hpp"

#include <cmath>
#include <cstdio>
#include <string_view>
#include "test_logging.hpp"

namespace
{

	namespace auto_aim = app::auto_aim;

	constexpr double kPi = 3.14159265358979323846;
	constexpr double kTwoPi = 2.0 * kPi;

	// ============================================================
	// 简单测试运行器
	// ============================================================

	class TestRunner
	{
	public:
		void begin(std::string_view name)
		{
			current_test_ = name;
			current_test_failed_ = false;

			std::printf("===== %.*s =====\n", static_cast<int>(name.size()), name.data());
		}

		void expect(bool condition, std::string_view message)
		{
			++check_count_;

			if(condition)
			{
				std::printf("[PASS] %.*s\n", static_cast<int>(message.size()), message.data());
				return;
			}

			++failure_count_;
			current_test_failed_ = true;

			std::printf("[FAIL] %.*s\n", static_cast<int>(message.size()), message.data());
		}

		void end()
		{
			std::printf("[%s] %.*s\n\n", current_test_failed_ ? "FAILED" : "PASSED",
			            static_cast<int>(current_test_.size()), current_test_.data());
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
		std::string_view current_test_;

		int check_count_ = 0;
		int failure_count_ = 0;

		bool current_test_failed_ = false;
	};

	bool near(double lhs, double rhs, double eps = 1e-9)
	{
		return std::abs(lhs - rhs) <= eps;
	}

	bool vector_near(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs, double eps = 1e-9)
	{
		return (lhs - rhs).norm() <= eps;
	}

	// ============================================================
	// 构造 helper
	// ============================================================

	auto_aim::TargetModelConfig deterministic_config()
	{
		auto_aim::TargetModelConfig c;
		c.translation_accel_variance = 1.0;
		c.yaw_accel_variance = 1.0;
		c.radius_random_walk_variance = 1.0;
		c.delta_radius_random_walk_variance = 1.0;
		c.delta_z_random_walk_variance = 1.0;
		return c;
	}

	auto_aim::ArmorObservation make_observation(double x, double y, double z, double yaw,
	                                            auto_aim::ArmorName name)
	{
		auto_aim::ArmorObservation o;
		o.color = auto_aim::ArmorColor::Red;
		o.name = name;
		o.type = auto_aim::ArmorType::Small;
		o.position_in_world = Eigen::Vector3d(x, y, z);
		o.armor_yaw_in_world = yaw;
		return o;
	}

	// 直接构造一个 TrackedTarget 快照（predicted_armors 只用于提供 armor_count）。
	auto_aim::TrackedTarget make_tracked_target(Eigen::Vector3d center, Eigen::Vector3d velocity,
	                                            double yaw, double yaw_rate, double radius,
	                                            double delta_radius, double delta_z, int armor_count)
	{
		auto_aim::TrackedTarget t;
		t.center_in_world = center;
		t.velocity_in_world = velocity;
		t.yaw = yaw;
		t.yaw_rate = yaw_rate;
		t.radius = radius;
		t.delta_radius = delta_radius;
		t.delta_z = delta_z;

		for(int i = 0; i < armor_count; ++i)
		{
			auto_aim::ArmorHypothesis h;
			h.armor_id = i;
			t.predicted_armors.push_back(h);
		}
		return t;
	}

	// 从 Target 状态构造等价 TrackedTarget（镜像 Tracker::make_snapshot）。
	auto_aim::TrackedTarget snapshot_from_target(const auto_aim::Target& target)
	{
		const Eigen::VectorXd& x = target.state();

		auto_aim::TrackedTarget t;
		t.center_in_world = Eigen::Vector3d(x(auto_aim::kStateX), x(auto_aim::kStateY),
		                                    x(auto_aim::kStateZ));
		t.velocity_in_world = Eigen::Vector3d(x(auto_aim::kStateVx), x(auto_aim::kStateVy),
		                                      x(auto_aim::kStateVz));
		t.yaw = x(auto_aim::kStateYaw);
		t.yaw_rate = x(auto_aim::kStateYawRate);
		t.radius = x(auto_aim::kStateRadius);
		t.delta_radius = x(auto_aim::kStateDeltaRadius);
		t.delta_z = x(auto_aim::kStateDeltaZ);
		t.predicted_armors = target.armor_hypotheses();
		return t;
	}

	// ============================================================
	// 测试用例
	// ============================================================

	void test_equivalence_dt0(TestRunner& runner)
	{
		runner.begin("Equivalence with Target at dt == 0");

		const double radius = 0.25;
		const double yaw = 0.7;

		auto o = make_observation(1.0, 2.0, 3.0, yaw, auto_aim::ArmorName::Four);

		Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(auto_aim::kTargetStateDim,
		                                              auto_aim::kTargetStateDim);

		auto_aim::Target target(o, radius, P0, deterministic_config());

		const auto snapshot = snapshot_from_target(target);
		const auto predicted = auto_aim::predict_vehicle(snapshot, 0.0);
		const auto hypotheses = auto_aim::armor_hypotheses(predicted);
		const auto& expected = target.armor_hypotheses();

		runner.expect(hypotheses.size() == expected.size(), "hypothesis count should match");

		for(std::size_t i = 0; i < hypotheses.size() && i < expected.size(); ++i)
		{
			runner.expect(hypotheses[i].armor_id == expected[i].armor_id, "armor_id should match");
			runner.expect(vector_near(hypotheses[i].position_in_world, expected[i].position_in_world,
			                          1e-12),
			              "position should match");
			runner.expect(near(hypotheses[i].yaw_in_world, expected[i].yaw_in_world, 1e-12),
			              "yaw should match");
		}

		runner.end();
	}

	void test_translation(TestRunner& runner)
	{
		runner.begin("Translation");

		const auto t = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 2.0, 3.0),
		                                   0.0, 0.0, 0.2, 0.0, 0.0, 4);
		const auto predicted = auto_aim::predict_vehicle(t, 2.0);

		runner.expect(vector_near(predicted.center, Eigen::Vector3d(2.0, 4.0, 6.0), 1e-12),
		              "center should translate by velocity * dt");

		runner.end();
	}

	void test_rotation(TestRunner& runner)
	{
		runner.begin("Rotation");

		// 正常：yaw += yaw_rate * dt。
		const auto t1 = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0,
		                                    1.0, 0.2, 0.0, 0.0, 4);
		const auto p1 = auto_aim::predict_vehicle(t1, 0.5);
		runner.expect(near(p1.yaw, 0.5, 1e-12), "yaw should advance by yaw_rate * dt");

		// 归一化：3.0 + 0.5 = 3.5 -> wrap 到 [-π, π)。
		const auto t2 = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 3.0,
		                                    1.0, 0.2, 0.0, 0.0, 4);
		const auto p2 = auto_aim::predict_vehicle(t2, 0.5);
		runner.expect(near(p2.yaw, 3.5 - kTwoPi, 1e-12), "yaw should wrap to [-pi, pi)");

		runner.end();
	}

	void test_alternating_radius_z(TestRunner& runner)
	{
		runner.begin("Alternating radius / z (armor_count == 4)");

		const auto t = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0,
		                                   0.0, 0.2, 0.1, 0.3, 4);
		const auto hs = auto_aim::armor_hypotheses(auto_aim::predict_vehicle(t, 0.0));

		runner.expect(hs.size() == 4, "should produce 4 hypotheses");

		// id0: theta=0, r=0.2, z=0 -> (-0.2, 0, 0)
		runner.expect(vector_near(hs[0].position_in_world, Eigen::Vector3d(-0.2, 0.0, 0.0), 1e-12),
		              "id0 position");
		runner.expect(near(hs[0].position_in_world.z(), 0.0, 1e-12), "id0 z");

		// id1: theta=π/2, r=0.3, z=0.3 -> (0, -0.3, 0.3)
		runner.expect(vector_near(hs[1].position_in_world, Eigen::Vector3d(0.0, -0.3, 0.3), 1e-12),
		              "id1 position");

		// id2: theta=π, r=0.2 -> (0.2, 0, 0)
		runner.expect(vector_near(hs[2].position_in_world, Eigen::Vector3d(0.2, 0.0, 0.0), 1e-12),
		              "id2 position");

		// id3: theta=3π/2, r=0.3, z=0.3 -> (0, 0.3, 0.3)
		runner.expect(vector_near(hs[3].position_in_world, Eigen::Vector3d(0.0, 0.3, 0.3), 1e-12),
		              "id3 position");

		runner.end();
	}

	void test_armor_indexing(TestRunner& runner)
	{
		runner.begin("Armor indexing (armor_count == 2 / 3)");

		// count == 2：无 alternate。
		const auto t2 = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0,
		                                    0.0, 0.2, 0.1, 0.3, 2);
		const auto hs2 = auto_aim::armor_hypotheses(auto_aim::predict_vehicle(t2, 0.0));
		runner.expect(hs2.size() == 2, "count 2 -> 2 hypotheses");
		runner.expect(near(hs2[0].yaw_in_world, 0.0, 1e-12), "count2 id0 yaw == 0");
		// wrap_angle 返回 [-π, π)：wrap(π) == -π。
		runner.expect(near(hs2[1].yaw_in_world, -kPi, 1e-12), "count2 id1 yaw == -pi");

		// count == 3（outpost）：无 alternate，theta 步进 2π/3。
		const auto t3 = make_tracked_target(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0,
		                                    0.0, 0.2, 0.1, 0.3, 3);
		const auto hs3 = auto_aim::armor_hypotheses(auto_aim::predict_vehicle(t3, 0.0));
		runner.expect(hs3.size() == 3, "count 3 -> 3 hypotheses");
		runner.expect(near(hs3[0].yaw_in_world, 0.0, 1e-12), "count3 id0 yaw == 0");
		runner.expect(near(hs3[1].yaw_in_world, kTwoPi / 3.0, 1e-12), "count3 id1 yaw");
		// wrap(4π/3) == -2π/3。
		runner.expect(near(hs3[2].yaw_in_world, -kTwoPi / 3.0, 1e-12), "count3 id2 yaw");

		runner.end();
	}

	void test_extrapolate(TestRunner& runner)
	{
		runner.begin("Signed extrapolation (extrapolate_vehicle)");

		const auto t = make_tracked_target(Eigen::Vector3d(1.0, 2.0, 3.0),
		                                   Eigen::Vector3d(0.5, 0.5, 0.5), 0.0, 1.0, 0.2, 0.0, 0.0,
		                                   4);

		// 正向：与 predict_vehicle 一致。
		const auto fwd = auto_aim::extrapolate_vehicle(t, 0.5);
		const auto ref = auto_aim::predict_vehicle(t, 0.5);
		runner.expect(near(fwd.center.x(), ref.center.x(), 1e-12), "forward center.x == predict");
		runner.expect(near(fwd.center.y(), ref.center.y(), 1e-12), "forward center.y == predict");
		runner.expect(near(fwd.center.z(), ref.center.z(), 1e-12), "forward center.z == predict");
		runner.expect(near(fwd.yaw, ref.yaw, 1e-12), "forward yaw == predict");

		// 负向：正确回推 center 与 yaw。
		const auto bwd = auto_aim::extrapolate_vehicle(t, -0.5);
		runner.expect(near(bwd.center.x(), 1.0 - 0.25, 1e-12), "backward center.x");
		runner.expect(near(bwd.center.y(), 2.0 - 0.25, 1e-12), "backward center.y");
		runner.expect(near(bwd.center.z(), 3.0 - 0.25, 1e-12), "backward center.z");
		runner.expect(near(bwd.yaw, -0.5, 1e-12), "backward yaw");

		runner.end();
	}

} // namespace

int main()
{
	test_logging::init("test_vehicle_prediction");
	std::printf("=== Vehicle Prediction Test Suite ===\n\n");

	TestRunner runner;

	test_equivalence_dt0(runner);
	test_translation(runner);
	test_rotation(runner);
	test_alternating_radius_z(runner);
	test_armor_indexing(runner);
	test_extrapolate(runner);

	runner.print_summary();

	if(runner.failure_count() != 0)
	{
		std::printf("=== Vehicle prediction tests failed ===\n");
		return 1;
	}

	std::printf("=== All vehicle prediction tests passed ===\n");
	return 0;
}
