//
// Created by tgu on 2026/4/14.
//

#include "tools/logger.hpp"
#include "tools/logger_config_loader.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr const char* MODULE = "TEST";

std::string dated_log_path() {
    const auto now = std::chrono::system_clock::now();
    const auto time_now = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&time_now, &local_time);

    std::ostringstream stream;
    stream << PROJECT_SOURCE_DIR "/data/logs/test_logger_"
           << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S") << ".log";
    return stream.str();
}

bool file_contains(const std::string& path, const std::string& expected) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    const std::string content(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());

    return content.find(expected) != std::string::npos;
}

}  // namespace

int main() {
    tools::LoggerConfig config;
    std::string config_error;

    if (!tools::load_logger_config(
	    PROJECT_SOURCE_DIR "/config/tools/logger_config.toml",
            config,
            &config_error)) {
        std::cerr << "logger config load failed: "
                  << config_error << '\n';
        return 1;
    }

    if (config.file_path != "data/logs/robocore.log") {
        std::cerr << "unexpected logger file path: "
                  << config.file_path << '\n';
        return 2;
    }

    // Do not pollute the formal runtime log during unit testing.
    config.file_path = PROJECT_SOURCE_DIR "/data/logs/test_logger.log";

    const std::string log_path = dated_log_path();
    std::error_code error_code;
    std::filesystem::remove(log_path, error_code);

    if (!tools::Logger::instance().init(config)) {
        std::cerr << "logger init failed\n";
        return 3;
    }

    const int rx_len = 64;
    const std::string device = "/dev/ttyACM0";

    LOG_DEBUG(MODULE, "rx len={}", rx_len);
    LOG_INFO(MODULE, "{} open success", device);
    LOG_WARN(MODULE, "crc check failed");
    LOG_ERROR(MODULE, "{} open failed: {}", device, "permission denied");

    tools::Logger::instance().flush();

    if (!file_contains(log_path, "[TEST] - [")) {
        std::cerr << "missing module or timestamp\n";
        return 4;
    }

#ifndef NDEBUG
    if (!file_contains(log_path, "[DEBUG] - rx len=64")) {
        std::cerr << "missing debug log\n";
        return 5;
    }
#endif

    if (!file_contains(log_path, "[INFO] - /dev/ttyACM0 open success")) {
        std::cerr << "missing info log\n";
        return 6;
    }

    if (!file_contains(log_path, "[WARN] - crc check failed")) {
        std::cerr << "missing warn log\n";
        return 7;
    }

    if (!file_contains(log_path,
            "[ERROR] - /dev/ttyACM0 open failed: permission denied")) {
        std::cerr << "missing error log\n";
        return 8;
    }

    tools::Logger::instance().shutdown();
    std::cout << "logger test passed\n";
    return 0;
}
