#pragma once

#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

namespace io
{

	struct UsbCameraConfig
	{
		std::string device = "/dev/v4l/by-id/usb-lihappe8_Corp._USB_2.0_Camera-video-index0";

		int width = 640;
		int height = 480;
		int fps = 30;

		// "MJPG" or "YUYV"
		std::string pixel_format = "MJPG";

		int buffer_size = 1;

		// 尝试关闭自动曝光。
		bool enable_manual_exposure = false;
		double exposure = 80.0;

		bool enable_manual_gain = false;
		double gain = 0.0;
	};

	struct CameraFrame
	{
		cv::Mat image;
		double timestamp_s = 0.0;
		std::uint64_t frame_id = 0;
	};

	class UsbCamera
	{
	public:
		explicit UsbCamera(UsbCameraConfig config);
		~UsbCamera();

		UsbCamera(const UsbCamera&) = delete;
		UsbCamera& operator=(const UsbCamera&) = delete;

		UsbCamera(UsbCamera&&) = delete;
		UsbCamera& operator=(UsbCamera&&) = delete;

		bool open();
		bool is_opened() const;
		bool read(CameraFrame& frame);
		void close();

		[[nodiscard]] const UsbCameraConfig& config() const;

	private:
		static int fourcc_from_string(const std::string& format);
		static double now_seconds();

	private:
		UsbCameraConfig config_;
		cv::VideoCapture cap_;
		std::uint64_t frame_id_ = 0;
	};

} // namespace io
