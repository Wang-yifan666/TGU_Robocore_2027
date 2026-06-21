//
// Created by tgu on 2026/4/14.
//

#ifndef TGU_ROBOCORE_2027_LOGGER_HPP
#define TGU_ROBOCORE_2027_LOGGER_HPP

#pragma once

#include <atomic>
#include <exception>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace tools
{

	enum class LogLevel
	{
		Debug = 0,
		Info,
		Warn,
		Error,
		Off
	};

	struct LoggerConfig
	{
		LogLevel level = LogLevel::Debug;
		bool enable_console = true;
		bool enable_file = false;
		bool flush_on_error = true;
		std::string file_path = "data/logs/robocore.log";
	};

	class Logger
	{
	public:
		static Logger& instance();

		/**
     * @brief 初始化日志模块。
     * @return 文件输出未启用或日志文件成功打开时返回 true，否则返回 false。
     */
		bool init(const LoggerConfig& config);

		/**
     * @brief 刷新并关闭日志文件。
     */
		void shutdown();

		/**
     * @brief 主动刷新日志文件。
     */
		void flush();

		void set_level(LogLevel level) noexcept;

		[[nodiscard]] LogLevel level() const noexcept;

		[[nodiscard]] bool should_log(LogLevel level) const noexcept;

		template<typename... Args>
		void log(LogLevel level, std::string_view module, std::format_string<Args...> format_string,
		         Args&&... args)
		{
			if(!should_log(level))
			{
				return;
			}

			try
			{
				write(level, module, std::format(format_string, std::forward<Args>(args)...));
			}
			catch(const std::exception& e)
			{
				write(LogLevel::Error, "LOGGER", std::string("log formatting failed: ") + e.what());
			}
			catch(...)
			{
				write(LogLevel::Error, "LOGGER", "log formatting failed: unknown exception");
			}
		}

		Logger(const Logger&) = delete;
		Logger& operator=(const Logger&) = delete;
		Logger(Logger&&) = delete;
		Logger& operator=(Logger&&) = delete;

	private:
		Logger() = default;
		~Logger();

		void write(LogLevel level, std::string_view module, std::string_view message);

		[[nodiscard]] std::string format_line(LogLevel level, std::string_view module,
		                                      std::string_view message) const;

		[[nodiscard]] static std::string_view level_to_string(LogLevel level) noexcept;

	private:
		std::atomic<LogLevel> level_{LogLevel::Debug};

		bool console_ = true;
		bool file_ = false;
		bool flush_on_error_ = true;

		std::ofstream ofs_;
		mutable std::mutex mutex_;
	};

} // namespace tools

#define ROBOCORE_LOG_IMPL(level_value, module, fmt, ...)                                       \
	do                                                                                         \
	{                                                                                          \
		auto& robocore_logger_instance = ::tools::Logger::instance();                          \
		if(robocore_logger_instance.should_log(level_value))                                   \
		{                                                                                      \
			robocore_logger_instance.log(level_value, module, fmt __VA_OPT__(, ) __VA_ARGS__); \
		}                                                                                      \
	} while(false)

#ifdef NDEBUG
#define LOG_DEBUG(module, fmt, ...) ((void)0)
#else
#define LOG_DEBUG(module, fmt, ...) \
	ROBOCORE_LOG_IMPL(::tools::LogLevel::Debug, module, fmt __VA_OPT__(, ) __VA_ARGS__)
#endif

#define LOG_INFO(module, fmt, ...) \
	ROBOCORE_LOG_IMPL(::tools::LogLevel::Info, module, fmt __VA_OPT__(, ) __VA_ARGS__)

#define LOG_WARN(module, fmt, ...) \
	ROBOCORE_LOG_IMPL(::tools::LogLevel::Warn, module, fmt __VA_OPT__(, ) __VA_ARGS__)

#define LOG_ERROR(module, fmt, ...) \
	ROBOCORE_LOG_IMPL(::tools::LogLevel::Error, module, fmt __VA_OPT__(, ) __VA_ARGS__)

#endif // TGU_ROBOCORE_2027_LOGGER_HPP
