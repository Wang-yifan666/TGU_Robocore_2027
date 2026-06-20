#include "logger.hpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace app
{
	namespace logger
	{
		std::shared_ptr<spdlog::logger> logger_ = nullptr;

		void set_logger()
		{
			auto file_name = fmt::format()
		}
	} // namespace logger
} // namespace app