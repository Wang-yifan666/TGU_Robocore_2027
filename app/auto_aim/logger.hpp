#ifndef TGU_ROBOCORE_2027_LOGGER_HPP
#define TGU_ROBOCORE_2027_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace app
{
	namespace logger
	{
		std::shared_ptr<spdlog::logger> logger();
	} // namespace logger
} // namespace app