//
// Created by tgu on 2026/4/14.
//

#include "logger.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

namespace tools
{

	Logger& Logger::instance()
	{
		static Logger instance;
		return instance;
	}

	Logger::~Logger()
	{
		shutdown();
	}

	bool Logger::init(const LoggerConfig& config)
	{
		std::lock_guard lock(mutex_);

		if(ofs_.is_open())
		{
			ofs_.flush();
			ofs_.close();
		}
		ofs_.clear();

		level_.store(config.level, std::memory_order_relaxed);
		console_ = config.enable_console;
		file_ = false;
		flush_on_error_ = config.flush_on_error;

		if(!config.enable_file)
		{
			return true;
		}

		if(config.file_path.empty())
		{
			std::cerr << "[LOGGER] log file path is empty\n";
			return false;
		}

		// 自动在文件名中插入当前日期时间，使每次运行生成独立的日志文件
		// 例如: "data/logs/robocore.log" → "data/logs/robocore_2026-06-22_01-59-33.log"
		const auto now = std::chrono::system_clock::now();
		const auto time_now = std::chrono::system_clock::to_time_t(now);
		std::tm local_time{};
		localtime_r(&time_now, &local_time);

		std::ostringstream date_stream;
		date_stream << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");

		const std::filesystem::path base_path(config.file_path);
		const std::string stem = base_path.stem().string();
		const std::string extension = base_path.extension().string();
		const std::string dated_filename = stem + '_' + date_stream.str() + extension;
		const std::filesystem::path log_path = base_path.parent_path() / dated_filename;
		const std::filesystem::path parent_path = log_path.parent_path();

		if(!parent_path.empty())
		{
			std::error_code error_code;
			std::filesystem::create_directories(parent_path, error_code);

			if(error_code)
			{
				std::cerr << "[LOGGER] failed to create log directory: " << parent_path.string()
				          << ", reason: " << error_code.message() << '\n';
				return false;
			}
		}

		ofs_.open(log_path, std::ios::out | std::ios::app);
		if(!ofs_.is_open())
		{
			std::cerr << "[LOGGER] failed to open log file: " << log_path.string() << '\n';
			return false;
		}

		file_ = true;
		return true;
	}

	void Logger::shutdown()
	{
		std::lock_guard lock(mutex_);

		if(ofs_.is_open())
		{
			ofs_.flush();
			ofs_.close();
		}

		ofs_.clear();
		file_ = false;
	}

	void Logger::flush()
	{
		std::lock_guard lock(mutex_);

		if(ofs_.is_open())
		{
			ofs_.flush();
		}

		std::cout.flush();
		std::cerr.flush();
	}

	void Logger::set_level(LogLevel level) noexcept
	{
		level_.store(level, std::memory_order_relaxed);
	}

	LogLevel Logger::level() const noexcept
	{
		return level_.load(std::memory_order_relaxed);
	}

	bool Logger::should_log(LogLevel level) const noexcept
	{
		const LogLevel current_level = this->level();

		if(level == LogLevel::Off || current_level == LogLevel::Off)
		{
			return false;
		}

		return static_cast<int>(level) >= static_cast<int>(current_level);
	}

	void Logger::write(LogLevel level, std::string_view module, std::string_view message)
	{
		if(!should_log(level))
		{
			return;
		}

		const std::string text = format_line(level, module, message);

		std::lock_guard lock(mutex_);

		if(console_)
		{
			std::ostream& stream = level >= LogLevel::Warn ? std::cerr : std::cout;
			stream << text << '\n';

			if(flush_on_error_ && level >= LogLevel::Error)
			{
				stream.flush();
			}
		}

		if(file_ && ofs_.is_open())
		{
			ofs_ << text << '\n';

			if(flush_on_error_ && level >= LogLevel::Error)
			{
				ofs_.flush();
			}
		}
	}

	std::string Logger::format_line(LogLevel level, std::string_view module,
	                                std::string_view message) const
	{
		using namespace std::chrono;

		const auto now = system_clock::now();
		const auto time = system_clock::to_time_t(now);
		const auto milliseconds =
		    duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

		std::tm local_time{};

#if defined(_WIN32)
		localtime_s(&local_time, &time);
#else
		localtime_r(&time, &local_time);
#endif

		const std::string_view source = module.empty() ? "UNKNOWN" : module;

		std::ostringstream stream;
		stream << '[' << source << "] - [" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << '.'
		       << std::setfill('0') << std::setw(3) << milliseconds << "] - ["
		       << level_to_string(level) << "] - " << message;

		return stream.str();
	}

	std::string_view Logger::level_to_string(LogLevel level) noexcept
	{
		switch(level)
		{
		case LogLevel::Debug:
			return "DEBUG";
		case LogLevel::Info:
			return "INFO";
		case LogLevel::Warn:
			return "WARN";
		case LogLevel::Error:
			return "ERROR";
		case LogLevel::Off:
			return "OFF";
		}

		return "UNKNOWN";
	}

} // namespace tools
