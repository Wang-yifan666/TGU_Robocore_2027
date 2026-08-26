/**
 * @file test_logging.hpp
 * @brief 测试日志初始化辅助（统一所有测试的 Logger 文件输出）。
 *
 * 每个测试的 main() 开头调用 test_logging::init("test_xxx")，
 * 日志会同时输出到控制台与 data/logs/test_xxx_<timestamp>.log。
 */

#ifndef TGU_ROBOCORE_2027_TEST_LOGGING_HPP
#define TGU_ROBOCORE_2027_TEST_LOGGING_HPP

#include <string>

#include "tools/logger.hpp"

namespace test_logging
{

	/**
	 * @brief 初始化测试日志：控制台 + 文件输出。
	 * @param name 测试名（用于生成日志文件名）。
	 * @return 同 tools::Logger::init 的返回值（文件输出失败时为 false）。
	 */
	inline bool init(const std::string& name)
	{
		tools::LoggerConfig config;
		config.level = tools::LogLevel::Debug;
		config.enable_console = true;
		config.enable_file = true;
		config.file_path = std::string(PROJECT_SOURCE_DIR) + "/data/logs/" + name + ".log";
		return tools::Logger::instance().init(config);
	}

} // namespace test_logging

#endif // TGU_ROBOCORE_2027_TEST_LOGGING_HPP
