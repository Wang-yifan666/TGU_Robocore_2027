// 帮助 logger 读取日志的配置

#ifndef TGU_ROBOCORE_2027_LOGGER_CONFIG_LOADER_HPP
#define TGU_ROBOCORE_2027_LOGGER_CONFIG_LOADER_HPP

#pragma once

#include <filesystem>
#include <string>

#include "tools/logger.hpp"

namespace tools {

inline constexpr const char* DEFAULT_LOGGER_CONFIG_PATH =
    "tools/config/logger_config.toml";

/**
 * @brief Load LoggerConfig from a TOML file.
 *
 * Missing configuration items keep the values already stored in config.
 * If parsing or validation fails, config is left unchanged.
 *
 * @param config_path TOML configuration file path.
 * @param config Output configuration and fallback values.
 * @param error_message Optional error description.
 * @return true if the file was parsed and validated successfully.
 */
[[nodiscard]] bool load_logger_config(
    const std::filesystem::path& config_path,
    LoggerConfig& config,
    std::string* error_message = nullptr);

}  // namespace tools

#endif  // TGU_ROBOCORE_2027_LOGGER_CONFIG_LOADER_HPP
