#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "io/usbcamera/usbcamera.hpp"
#include "io/usbcamera/usbcamera_config.hpp"
#include "tools/logger.hpp"

int main()
{
	tools::LoggerConfig logger_config;
	logger_config.level = tools::LogLevel::Debug;
	logger_config.enable_console = true;
	logger_config.enable_file = false;
	tools::Logger::instance().init(logger_config);

	io::UsbCameraConfig config;

	// 从 TOML 配置文件加载 USB 摄像头参数
	const std::filesystem::path config_path =
	    std::filesystem::path(PROJECT_SOURCE_DIR) / "config/camera.toml";

	std::string error_msg;
	if(!io::load_usb_camera_config(config_path, config, &error_msg))
	{
		LOG_WARN("TestUsbCamera", "Failed to load config: {}, using defaults", error_msg);
	}
	else
	{
		LOG_INFO("TestUsbCamera", "Loaded config from {}", config_path.string());
		LOG_INFO("TestUsbCamera", "  device                = {}", config.device);
		LOG_INFO("TestUsbCamera", "  width                 = {}", config.width);
		LOG_INFO("TestUsbCamera", "  height                = {}", config.height);
		LOG_INFO("TestUsbCamera", "  fps                   = {}", config.fps);
		LOG_INFO("TestUsbCamera", "  pixel_format          = {}", config.pixel_format);
		LOG_INFO("TestUsbCamera", "  buffer_size           = {}", config.buffer_size);
		LOG_INFO("TestUsbCamera", "  enable_manual_exposure= {}", config.enable_manual_exposure);
		LOG_INFO("TestUsbCamera", "  exposure              = {}", config.exposure);
		LOG_INFO("TestUsbCamera", "  enable_manual_gain    = {}", config.enable_manual_gain);
		LOG_INFO("TestUsbCamera", "  gain                  = {}", config.gain);
	}

	io::UsbCamera camera(config);

	if(!camera.open())
	{
		std::cerr << "Failed to open usb camera" << std::endl;
		return -1;
	}

	io::CameraFrame frame;

	while(true)
	{
		if(!camera.read(frame))
		{
			continue;
		}

		cv::Mat show = frame.image.clone();

		const std::string info = "frame_id: " + std::to_string(frame.frame_id)
		    + "  timestamp: " + std::to_string(frame.timestamp_s);

		cv::putText(show, info, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
		            cv::Scalar(0, 255, 0), 2);

		cv::imshow("usb_camera", show);

		const int key = cv::waitKey(1);
		if(key == 27 || key == 'q')
		{
			break;
		}

		if(key == 's')
		{
			const std::string filename = "usb_camera_capture.jpg";
			cv::imwrite(filename, frame.image);
			std::cout << "Saved " << filename << std::endl;
		}
	}

	camera.close();
	tools::Logger::instance().shutdown();

	return 0;
}
