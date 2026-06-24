#include "io/usbcamera/usbcamera.hpp"

#include <chrono>
#include <iostream>

#include "tools/logger.hpp"

namespace io
{

	UsbCamera::UsbCamera(UsbCameraConfig config): config_(std::move(config)) {}

	UsbCamera::~UsbCamera()
	{
		close();
	}

	bool UsbCamera::open()
	{
		if(cap_.isOpened())
		{
			return true;
		}

		cap_.open(config_.device, cv::CAP_V4L2);

		if(!cap_.isOpened())
		{
			LOG_ERROR("UsbCamera", "Failed to open camera: {}", config_.device);
			return false;
		}

		const int fourcc = fourcc_from_string(config_.pixel_format);
		if(fourcc != 0)
		{
			cap_.set(cv::CAP_PROP_FOURCC, fourcc);
		}

		cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
		cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
		cap_.set(cv::CAP_PROP_FPS, config_.fps);
		cap_.set(cv::CAP_PROP_BUFFERSIZE, config_.buffer_size);

		if(config_.enable_manual_exposure)
		{
			// V4L2 下 OpenCV 的自动曝光取值比较玄学：
			// 常见情况 0.25 = manual, 0.75 = auto。
			// 如果不生效，后面改用 v4l2-ctl 或直接 V4L2 API。
			cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
			cap_.set(cv::CAP_PROP_EXPOSURE, config_.exposure);
		}

		if(config_.enable_manual_gain)
		{
			cap_.set(cv::CAP_PROP_GAIN, config_.gain);
		}

		const int actual_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
		const int actual_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
		const double actual_fps = cap_.get(cv::CAP_PROP_FPS);
		const int actual_fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));

		char fourcc_text[] = {
		    static_cast<char>(actual_fourcc & 0xFF),
		    static_cast<char>((actual_fourcc >> 8) & 0xFF),
		    static_cast<char>((actual_fourcc >> 16) & 0xFF),
		    static_cast<char>((actual_fourcc >> 24) & 0xFF),
		    '\0',
		};

		LOG_INFO("UsbCamera", "Opened camera: device={}, resolution={}x{}, fps={}, fourcc={}",
		         config_.device, actual_width, actual_height, actual_fps, fourcc_text);

		return true;
	}

	bool UsbCamera::is_opened() const
	{
		return cap_.isOpened();
	}

	bool UsbCamera::read(CameraFrame& frame)
	{
		if(!cap_.isOpened())
		{
			LOG_ERROR("UsbCamera", "read() called before camera opened");
			return false;
		}

		cv::Mat image;
		if(!cap_.read(image))
		{
			LOG_WARN("UsbCamera", "Failed to read frame");
			return false;
		}

		if(image.empty())
		{
			LOG_WARN("UsbCamera", "Empty frame");
			return false;
		}

		frame.image = image;
		frame.timestamp_s = now_seconds();
		frame.frame_id = frame_id_++;

		return true;
	}

	void UsbCamera::close()
	{
		if(cap_.isOpened())
		{
			cap_.release();
			LOG_INFO("UsbCamera", "Camera closed");
		}
	}

	const UsbCameraConfig& UsbCamera::config() const
	{
		return config_;
	}

	int UsbCamera::fourcc_from_string(const std::string& format)
	{
		if(format == "MJPG")
		{
			return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		}

		if(format == "YUYV")
		{
			return cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
		}

		LOG_WARN("UsbCamera", "Unknown pixel format: {}, skip FOURCC setting", format);
		return 0;
	}

	double UsbCamera::now_seconds()
	{
		using clock = std::chrono::steady_clock;
		const auto now = clock::now().time_since_epoch();
		return std::chrono::duration<double>(now).count();
	}

} // namespace io
