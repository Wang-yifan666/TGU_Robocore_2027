// 输入：
// 1. 装甲板四个图像点
// 2. 装甲板类型：small / large
// 3. 相机内参 camera_matrix
// 4. 畸变参数 dist_coeffs

// 输出：
// 1. rvec
// 2. tvec
// 3. distance
// 4. yaw
// 5. pitch
// 6. roll

#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_SOLVER_HPP

#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace app::auto_aim
{

}