//
#include "logger_config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "tools/tomlpp.hpp"

namespace tools
{

	namespace
	{

		std::string to_lower(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
				return static_cast<char>(std::tolower(character));
			});

			return value;
		}

		std::optional<LogLevel> parse_log_level(std::string value)
		{
			value = to_lower(std::move(value));

			if(value == "debug")
			{
				return LogLevel::Debug;
			}
			if(value == "info")
			{
				return LogLevel::Info;
			}
			if(value == "warn" || value == "warning")
			{
				return LogLevel::Warn;
			}
			if(value == "error")
			{
				return LogLevel::Error;
			}
			if(value == "off")
			{
				return LogLevel::Off;
			}

			return std::nullopt;
		}

		void set_error_message(std::string* error_message, std::string message)
		{
			if(error_message != nullptr)
			{
				*error_message = std::move(message);
			}
		}

	} // namespace

	bool load_logger_config(const std::filesystem::path& config_path, LoggerConfig& config,
	                        std::string* error_message)
	{
		LoggerConfig candidate = config;

		try
		{
			const toml::table table = toml::parse_file(config_path.string());
			const auto logger_table = table["logger"];

			if(!logger_table.is_table())
			{
				set_error_message(error_message,
				                  "missing [logger] table in " + config_path.string());
				return false;
			}

			if(const auto level_text = logger_table["level"].value<std::string>())
			{
				const auto parsed_level = parse_log_level(*level_text);

				if(!parsed_level.has_value())
				{
					set_error_message(error_message, "invalid logger.level: " + *level_text);
					return false;
				}

				candidate.level = *parsed_level;
			}

			candidate.enable_console =
			    logger_table["enable_console"].value_or(candidate.enable_console);

			candidate.enable_file = logger_table["enable_file"].value_or(candidate.enable_file);

			candidate.flush_on_error =
			    logger_table["flush_on_error"].value_or(candidate.flush_on_error);

			if(const auto file_path = logger_table["file_path"].value<std::string>())
			{
				candidate.file_path = *file_path;
			}

			if(candidate.enable_file && candidate.file_path.empty())
			{
				set_error_message(error_message,
				                  "logger.file_path cannot be empty when file output is enabled");
				return false;
			}

			config = std::move(candidate);

			if(error_message != nullptr)
			{
				error_message->clear();
			}

			return true;
		}
		catch(const toml::parse_error& error)
		{
			std::ostringstream stream;
			stream << "failed to parse " << config_path.string() << ": " << error.description();

			set_error_message(error_message, stream.str());
			return false;
		}
		catch(const std::exception& error)
		{
			set_error_message(error_message,
			                  "failed to load " + config_path.string() + ": " + error.what());
			return false;
		}
	}

} // namespace tools
