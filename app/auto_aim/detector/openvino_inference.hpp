/**
 * @file openvino_inference.hpp
 * @brief OpenVINO YOLOv5 推理后端：将 cv::Mat 解码为 std::vector<RawDetection>。
 *
 * 职责严格限定为：
 * - 模型加载与编译
 * - 图像 letterbox 预处理（BGR -> 信箱填充）
 * - OpenVINO 推理
 * - 输出 tensor 解码（每行 22 字段）
 * - 网络坐标映射回原图
 * - 生成 RawDetection
 *
 * 不负责：创建 Armor、判断敌方颜色、按 Armor 类型过滤、NMS、相机/串口/Tracker/Solver。
 *
 * 关键点顺序说明（保持与 sp_vision_25 YOLOV5 完全一致）：
 * 输出 tensor 每行字段为：
 *   col[0..1] = 关键点0 x/y
 *   col[2..3] = 关键点1 x/y
 *   col[4..5] = 关键点2 x/y
 *   col[6..7] = 关键点3 x/y
 *   col[8]     = objectness logit（需 sigmoid）
 *   col[9..12]  = 颜色 one-hot（4 类）
 *   col[13..21] = 编号 one-hot（9 类）
 *
 * 旧代码将原始字段重排为 Armor 所需的 [左上, 右上, 右下, 左下]：
 *   keypoints = { {col0,col1}, {col6,col7}, {col4,col5}, {col2,col3} }
 * （0-3 为左边；1-2 为右边；0-1 为上边；3-2 为下边）
 */
#ifndef TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_OPENVINO_INFERENCE_HPP
#define TGU_ROBOCORE_2027_APP_AUTO_AIM_DETECTOR_OPENVINO_INFERENCE_HPP

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "app/auto_aim/detector/inference.hpp"

namespace app::auto_aim::detector_detail
{

	/// YOLOv5 输出 tensor 每行字段数（8 个关键点坐标 + 1 objectness + 4 颜色 + 9 编号）。
	inline constexpr std::size_t kYoloV5RowWidth = 22;

	/**
	 * @brief 计算 letterbox 缩放比例（取宽高两个方向的最小缩放）。
	 * @param src_h 原图高。
	 * @param src_w 原图宽。
	 * @param dst_h 目标高（模型输入高）。
	 * @param dst_w 目标宽（模型输入宽）。
	 * @return 统一缩放比例。
	 */
	double letterbox_scale(int src_h, int src_w, int dst_h, int dst_w);

	/**
	 * @brief 根据 letterbox 缩放比例计算缩放后的尺寸（宽、高均向下取整为 int，与旧行为一致）。
	 * @param src_h 原图高。
	 * @param src_w 原图宽。
	 * @param scale letterbox 缩放比例。
	 * @return 缩放后尺寸（宽在前、高在后，即 cv::Size）。
	 */
	cv::Size letterbox_size(int src_h, int src_w, double scale);

	/**
	 * @brief 将网络坐标映射回原图坐标（左上角对齐，无居中偏移，故只需除以 scale）。
	 * @param point 网络坐标。
	 * @param scale letterbox 缩放比例。
	 * @return 原图坐标。
	 */
	cv::Point2f map_point_to_original(const cv::Point2f& point, double scale);

	/**
	 * @brief 单行解码结果。
	 */
	struct DecodedRow
	{
		RawDetection detection;
		bool accepted = false;
	};

	/**
	 * @brief 解码一行输出（22 个 float），并按置信度阈值做 early reject。
	 *
	 * 关键点顺序固定复现旧代码的 {col0/1, col6/7, col4/5, col2/3}。
	 *
	 * @param row 指向输出 tensor 一行的 22 个 float。
	 * @param scale letterbox 缩放比例。
	 * @param confidence_threshold objectness 阈值，低于该值则 accepted=false。
	 * @return DecodedRow；非法（NaN/Inf 关键点或置信度、非有限类别分数）时 accepted=false。
	 */
	DecodedRow decode_row(const float* row, double scale, float confidence_threshold);

} // namespace app::auto_aim::detector_detail

namespace app::auto_aim
{

	/**
	 * @brief 基于 OpenVINO 的 YOLOv5 推理后端。
	 */
	class OpenVINOInference final : public Inference
	{
	public:
		/**
		 * @param model_path OpenVINO IR .xml 模型路径。
		 * @param device 推理设备，例如 "CPU" / "GPU"。
		 * @param confidence_threshold objectness 置信度阈值，用于内存内 early reject。
		 *        （与 DetectorConfig::confidence_threshold 同一来源）
		 */
		OpenVINOInference(std::string model_path, std::string device, float confidence_threshold);

		[[nodiscard]] bool is_ready() const noexcept override;

		std::vector<RawDetection> infer(const cv::Mat& image) override;

	private:
		std::string model_path_;
		std::string device_;
		float confidence_threshold_ = 0.5F;

		ov::Core core_;
		ov::CompiledModel compiled_model_;

		std::size_t input_h_ = 0;
		std::size_t input_w_ = 0;

		bool ready_ = false;
	};

} // namespace app::auto_aim

#endif