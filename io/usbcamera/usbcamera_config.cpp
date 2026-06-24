#include "io/usbcamera/usbcamera_config.hpp"

#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "tools/tomlpp.hpp"

namespace io
{

namespace
{

void set_error_message(std::string* error_message, std::string message)
{
	if(error_message != nullptr)
	{
		*error_message = std::move(message);
	}
}

} // namespace

bool load_usb_camera_config(const std::filesystem::path& config_path,
                            UsbCameraConfig& config,
                            std::string* error_message)
{
	UsbCameraConfig candidate = config;

	try
	{
		const toml::table table = toml::parse_file(config_path.string());

		const auto usb_camera_node = table["usb_camera"];
		if(!usb_camera_node.is_table())
		{
			set_error_message(error_message,
			                  "missing [usb_camera] table in " + config_path.string());
			return false;
		}

		const auto& usb_camera_table = *usb_camera_node.as_table();

		// ---- device (string, must not be empty) ----
		{
			const auto node = usb_camera_table["device"];
			if(node)
			{
				const auto val = node.value<std::string>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.device must be a string");
					return false;
				}
				if(val->empty())
				{
					set_error_message(error_message,
					                  "usb_camera.device cannot be empty");
					return false;
				}
				candidate.device = *val;
			}
		}

		// ---- width (int, > 0) ----
		{
			const auto node = usb_camera_table["width"];
			if(node)
			{
				const auto val = node.value<int>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.width must be an integer");
					return false;
				}
				if(*val <= 0)
				{
					set_error_message(error_message,
					                  "usb_camera.width must be > 0, got " + std::to_string(*val));
					return false;
				}
				candidate.width = *val;
			}
		}

		// ---- height (int, > 0) ----
		{
			const auto node = usb_camera_table["height"];
			if(node)
			{
				const auto val = node.value<int>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.height must be an integer");
					return false;
				}
				if(*val <= 0)
				{
					set_error_message(error_message,
					                  "usb_camera.height must be > 0, got " + std::to_string(*val));
					return false;
				}
				candidate.height = *val;
			}
		}

		// ---- fps (int, > 0) ----
		{
			const auto node = usb_camera_table["fps"];
			if(node)
			{
				const auto val = node.value<int>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.fps must be an integer");
					return false;
				}
				if(*val <= 0)
				{
					set_error_message(error_message,
					                  "usb_camera.fps must be > 0, got " + std::to_string(*val));
					return false;
				}
				candidate.fps = *val;
			}
		}

		// ---- pixel_format (string, "MJPG" or "YUYV") ----
		{
			const auto node = usb_camera_table["pixel_format"];
			if(node)
			{
				const auto val = node.value<std::string>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.pixel_format must be a string");
					return false;
				}
				if(*val != "MJPG" && *val != "YUYV")
				{
					set_error_message(error_message,
					                  "usb_camera.pixel_format must be \"MJPG\" or \"YUYV\", got \""
					                      + *val + "\"");
					return false;
				}
				candidate.pixel_format = *val;
			}
		}

		// ---- buffer_size (int, > 0) ----
		{
			const auto node = usb_camera_table["buffer_size"];
			if(node)
			{
				const auto val = node.value<int>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.buffer_size must be an integer");
					return false;
				}
				if(*val <= 0)
				{
					set_error_message(error_message,
					                  "usb_camera.buffer_size must be > 0, got "
					                      + std::to_string(*val));
					return false;
				}
				candidate.buffer_size = *val;
			}
		}

		// ---- enable_manual_exposure (bool) ----
		{
			const auto node = usb_camera_table["enable_manual_exposure"];
			if(node)
			{
				const auto val = node.value<bool>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.enable_manual_exposure must be a boolean");
					return false;
				}
				candidate.enable_manual_exposure = *val;
			}
		}

		// ---- exposure (double) ----
		{
			const auto node = usb_camera_table["exposure"];
			if(node)
			{
				const auto val = node.value<double>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.exposure must be a floating-point number");
					return false;
				}
				candidate.exposure = *val;
			}
		}

		// ---- enable_manual_gain (bool) ----
		{
			const auto node = usb_camera_table["enable_manual_gain"];
			if(node)
			{
				const auto val = node.value<bool>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.enable_manual_gain must be a boolean");
					return false;
				}
				candidate.enable_manual_gain = *val;
			}
		}

		// ---- gain (double) ----
		{
			const auto node = usb_camera_table["gain"];
			if(node)
			{
				const auto val = node.value<double>();
				if(!val)
				{
					set_error_message(error_message,
					                  "usb_camera.gain must be a floating-point number");
					return false;
				}
				candidate.gain = *val;
			}
		}

		config = std::move(candidate);

		if(error_message != nullptr)
		{
			error_message->clear();
		}

		return true;
	}
	catch(const toml::parse_error& e)
	{
		std::ostringstream oss;
		oss << "failed to parse " << config_path.string() << ": " << e.description();
		set_error_message(error_message, oss.str());
		return false;
	}
	catch(const std::exception& e)
	{
		set_error_message(error_message,
		                  "failed to load " + config_path.string() + ": " + e.what());
		return false;
	}
}

} // namespace io
