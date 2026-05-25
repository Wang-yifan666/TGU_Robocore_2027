/**
 * @file solver.hpp
 * @brief PnP 求解器，计算装甲板相对于云台/世界的空间位置与姿态。
 *
 * 职责范围（待实现）：
 * - 通过 PnP 求解装甲板的旋转向量 rvec 和平移向量 tvec
 * - 计算距离 distance 以及欧拉角 yaw / pitch / roll
 * - 支持云台坐标系与世界坐标系的转换
 */

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP

#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace app::auto_aim
{

}
