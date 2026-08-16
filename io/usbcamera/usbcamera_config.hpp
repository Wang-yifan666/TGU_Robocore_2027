#pragma once

#include <filesystem>
#include <string>

#include "io/usbcamera/usbcamera.hpp"

namespace io
{

	/**
 * @brief Load UsbCameraConfig from a TOML configuration file.
 *
 * Reads the [usb_camera] table. Missing fields are left at their default values.
 * If a field exists but has a type mismatch, or validation fails, returns false
 * and sets error_message.
 *
 * @param config_path Path to the TOML configuration file.
 * @param config      Output config; existing values serve as fallback defaults.
 * @param error_message Optional output for a human-readable error description.
 * @return true if the file was parsed, validated, and the config was updated.
 */
	[[nodiscard]] bool load_usb_camera_config(const std::filesystem::path& config_path,
	                                          UsbCameraConfig& config,
	                                          std::string* error_message = nullptr);

} // namespace io
