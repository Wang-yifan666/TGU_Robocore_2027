#include "app/auto_aim/armor.hpp"
#include "tools/tomlpp.hpp"

#include <cstdio> 

// 从 config 加载测试参数
namespace
{

	// Lightbar 构造参数
	float LB_CENTER_X, LB_CENTER_Y, LB_SIZE_W, LB_SIZE_H, LB_ANGLE;

	// 左右灯条间距
	float ARMOR_LR_GAP;

	// class_id 测试
	int CLASS_ID_ONE_SMALL, CLASS_ID_THREE_BIG;

	// keypoints 定义
	std::vector<cv::Point2f> KP_RECT, KP_OFFSET, KP_YOLO, KP_SMALL;
	cv::Point2f OFFSET;

	// 期望值
	cv::Point2f EXPECT_CENTER_LB, EXPECT_CENTER_CLASS, EXPECT_CENTER_OFFSET;

	void load_config()
	{
		auto config =
		    toml::parse_file(std::string(PROJECT_SOURCE_DIR) + "/config/test/test_armor.toml");

		// Lightbar
		LB_CENTER_X = static_cast<float>(config["lightbar"]["center_x"].value_or(100.0));
		LB_CENTER_Y = static_cast<float>(config["lightbar"]["center_y"].value_or(100.0));
		LB_SIZE_W = static_cast<float>(config["lightbar"]["size_w"].value_or(10.0));
		LB_SIZE_H = static_cast<float>(config["lightbar"]["size_h"].value_or(40.0));
		LB_ANGLE = static_cast<float>(config["lightbar"]["angle"].value_or(0.0));

		// Armor
		ARMOR_LR_GAP = static_cast<float>(config["armor"]["lr_gap"].value_or(60.0));

		// class_id 测试
		CLASS_ID_ONE_SMALL = config["class_id_test"]["one_small"].value_or(3);
		CLASS_ID_THREE_BIG = config["class_id_test"]["three_big"].value_or(29);

		// keypoints
		auto load_kp = [](const toml::array& arr) -> std::vector<cv::Point2f> {
			std::vector<cv::Point2f> pts;
			for(const auto& elem: arr)
			{
				auto pt_arr = elem.as_array();
				if(pt_arr && pt_arr->size() >= 2)
				{
					float x = static_cast<float>((*pt_arr)[0].value_or(0.0));
					float y = static_cast<float>((*pt_arr)[1].value_or(0.0));
					pts.emplace_back(x, y);
				}
			}
			return pts;
		};

		auto kp_rect_arr = config["keypoints"]["rect"].as_array();
		if(kp_rect_arr)
			KP_RECT = load_kp(*kp_rect_arr);

		auto kp_offset_arr = config["keypoints"]["offset_kp"].as_array();
		if(kp_offset_arr)
			KP_OFFSET = load_kp(*kp_offset_arr);

		auto kp_yolo_arr = config["keypoints"]["yolo"].as_array();
		if(kp_yolo_arr)
			KP_YOLO = load_kp(*kp_yolo_arr);

		auto kp_small_arr = config["keypoints"]["small"].as_array();
		if(kp_small_arr)
			KP_SMALL = load_kp(*kp_small_arr);

		// offset
		auto offset_arr = config["keypoints"]["offset"].as_array();
		if(offset_arr && offset_arr->size() >= 2)
		{
			OFFSET.x = static_cast<float>((*offset_arr)[0].value_or(0.0));
			OFFSET.y = static_cast<float>((*offset_arr)[1].value_or(0.0));
		}

		// 期望值
		auto load_pt = [](const toml::array& arr) -> cv::Point2f {
			if(arr.size() >= 2)
			{
				return {static_cast<float>(arr[0].value_or(0.0)),
				        static_cast<float>(arr[1].value_or(0.0))};
			}
			return {};
		};

		auto center_lb_arr = config["expected"]["center_lb"].as_array();
		if(center_lb_arr)
			EXPECT_CENTER_LB = load_pt(*center_lb_arr);

		auto center_class_arr = config["expected"]["center_class"].as_array();
		if(center_class_arr)
			EXPECT_CENTER_CLASS = load_pt(*center_class_arr);

		auto center_offset_arr = config["expected"]["center_offset"].as_array();
		if(center_offset_arr)
			EXPECT_CENTER_OFFSET = load_pt(*center_offset_arr);
	}

} // namespace

// ==================== 测试函数 ====================

void TestLightbar()
{
	printf("===== Test Lightbar =====\n");

	cv::RotatedRect rr(cv::Point2f(LB_CENTER_X, LB_CENTER_Y), cv::Size2f(LB_SIZE_W, LB_SIZE_H),
	                   LB_ANGLE);
	auto_aim::Lightbar lb(rr, 0);

	printf("id: %zu\n", lb.id);
	printf("center: [%.0f, %.0f]\n", lb.center.x, lb.center.y);
	printf("top: [%.0f, %.0f]\n", lb.top.x, lb.top.y);
	printf("bottom: [%.0f, %.0f]\n", lb.bottom.x, lb.bottom.y);
	printf("length: %.1f\n", lb.length);
	printf("width: %.1f\n", lb.width);
	printf("angle: %.4f\n", lb.angle);
	printf("ratio: %.1f\n", lb.ratio);

	bool pass = (std::abs(lb.length - LB_SIZE_H) < 1e-3 && std::abs(lb.width - LB_SIZE_W) < 1e-3
	             && std::abs(lb.angle - CV_PI / 2.0) < 1e-3);
	printf("[%s] Lightbar geometry test\n\n", pass ? "PASS" : "FAIL");
}

void TestArmorFromLightbars()
{
	printf("===== Test Armor from Lightbars =====\n");

	cv::RotatedRect left_rr(cv::Point2f(LB_CENTER_X, LB_CENTER_Y), cv::Size2f(LB_SIZE_W, LB_SIZE_H),
	                        LB_ANGLE),
	    right_rr(cv::Point2f(LB_CENTER_X + ARMOR_LR_GAP, LB_CENTER_Y),
	             cv::Size2f(LB_SIZE_W, LB_SIZE_H), LB_ANGLE);
	auto_aim::Lightbar left_lb(left_rr, 0), right_lb(right_rr, 1);
	left_lb.color = right_lb.color = auto_aim::red;

	auto_aim::Armor armor(left_lb, right_lb);

	printf("color: %d\n", armor.color);
	printf("center: [%.0f, %.0f]\n", armor.center.x, armor.center.y);
	printf("ratio: %.1f\n", armor.ratio);
	printf("side_ratio: %.1f\n", armor.side_ratio);
	printf("rectangular_error: %.8f\n", armor.rectangular_error);

	bool pass = (cv::norm(armor.center - EXPECT_CENTER_LB) < 1e-3 && armor.color == auto_aim::red
	             && std::abs(armor.ratio - ARMOR_LR_GAP / LB_SIZE_H) < 1e-3
	             && std::abs(armor.side_ratio - 1.0) < 1e-3);
	printf("[%s] Armor from Lightbars test\n\n", pass ? "PASS" : "FAIL");
}

void TestArmorFromClassId()
{
	printf("===== Test Armor from class_id =====\n");

	auto_aim::Armor armor(CLASS_ID_ONE_SMALL, 0.85f, cv::Rect(50, 50, 60, 40),
	                      {KP_RECT.data(), KP_RECT.data() + 4});

	printf("class_id: %d\n", armor.class_id);
	printf("color: %d (expected blue=1)\n", armor.color);
	printf("name: %d (expected one=0)\n", armor.name);
	printf("type: %d (expected small=1)\n", armor.type);
	printf("center: [%.0f, %.0f]\n", armor.center.x, armor.center.y);

	bool pass =
	    (armor.color == auto_aim::blue && armor.name == auto_aim::one
	     && armor.type == auto_aim::small && cv::norm(armor.center - EXPECT_CENTER_CLASS) < 1e-3);
	printf("[%s] Armor from class_id test\n\n", pass ? "PASS" : "FAIL");
}

void TestArmorFromClassIdWithOffset()
{
	printf("===== Test Armor from class_id with offset =====\n");

	auto_aim::Armor armor(CLASS_ID_THREE_BIG, 0.9f, cv::Rect(100, 100, 80, 50),
	                      {KP_OFFSET.data(), KP_OFFSET.data() + 4}, OFFSET);

	for(size_t i = 0; i < armor.points.size(); i++)
		printf("  point[%zu]: [%.0f, %.0f]\n", i, armor.points[i].x, armor.points[i].y);

	bool pass =
	    (armor.color == auto_aim::blue && armor.name == auto_aim::three
	     && armor.type == auto_aim::big && cv::norm(armor.center - EXPECT_CENTER_OFFSET) < 1e-3);
	printf("[%s] Armor from class_id with offset test\n\n", pass ? "PASS" : "FAIL");
}

void TestArmorFromYoloId()
{
	printf("===== Test Armor from Yolo id =====\n");

	cv::Rect yolo_box(200, 200, 40, 30);
	auto_aim::Armor armor(0, 0, 0.75f, yolo_box, {KP_YOLO.data(), KP_YOLO.data() + 4});

	printf("color: %d (expected blue=1)\n", armor.color);
	printf("name: %d (expected sentry=5)\n", armor.name);
	printf("type: %d (expected small=1)\n", armor.type);

	bool pass = (armor.color == auto_aim::blue && armor.name == auto_aim::sentry
	             && armor.type == auto_aim::small);

	auto_aim::Armor armor2(1, 1, 0.8f, yolo_box, {KP_YOLO.data(), KP_YOLO.data() + 4});
	printf("armor2 color: %d (expected red=0)\n", armor2.color);
	printf("armor2 name: %d (expected one=0)\n", armor2.name);

	pass = pass && armor2.color == auto_aim::red && armor2.name == auto_aim::one;
	printf("[%s] Armor from Yolo id test\n\n", pass ? "PASS" : "FAIL");
}

void TestInvalidClassId()
{
	printf("===== Test Invalid class_id =====\n");

	auto_aim::Armor armor(-1, 0.0f, cv::Rect(), {KP_SMALL.data(), KP_SMALL.data() + 4});

	printf("color: %d (expected blue=1)\n", armor.color);
	printf("name: %d (expected not_armor=8)\n", armor.name);
	printf("type: %d (expected small=1)\n", armor.type);

	bool pass = (armor.color == auto_aim::blue && armor.name == auto_aim::not_armor
	             && armor.type == auto_aim::small);
	printf("[%s] Invalid class_id test\n\n", pass ? "PASS" : "FAIL");
}

int main()
{
	load_config();

	printf("=== Armor Module Test Suite ===\n\n");

	TestLightbar();
	TestArmorFromLightbars();
	TestArmorFromClassId();
	TestArmorFromClassIdWithOffset();
	TestArmorFromYoloId();
	TestInvalidClassId();

	printf("=== All tests completed ===\n");
	return 0;
}
