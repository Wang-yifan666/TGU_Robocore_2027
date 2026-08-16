#include "app/auto_aim/solver_config.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "tools/logger.hpp"
#include "tools/tomlpp.hpp"

namespace app::auto_aim
{

	namespace
	{

		constexpr std::string_view kLogModule = "SOLVER_CONFIG";

		bool is_finite(double value)
		{
			return std::isfinite(value);
		}

		/**
		 * @brief 读取 toml 数组为 double 向量。
		 *
		 * 非 numeric 元素（例如字符串）返回 std::nullopt，
		 * 不允许静默 fallback 到 0。
		 */
		std::optional<std::vector<double>> read_double_array(const toml::array* array)
		{
			if(array == nullptr)
			{
				return std::nullopt;
			}

			std::vector<double> values;
			values.reserve(array->size());

			for(const auto& element: *array)
			{
				const auto value = element.value<double>();
				if(!value)
				{
					return std::nullopt;
				}

				values.push_back(*value);
			}

			return values;
		}

		bool load_camera_matrix(const toml::table& camera_table, SolverConfig& config)
		{
			const auto values =
			    read_double_array(camera_table["camera_matrix"].as_array());

			if(!values)
			{
				LOG_ERROR(kLogModule, "camera_matrix must be an array of numeric values");
				return false;
			}

			if(values->size() != 9 || !std::all_of(values->begin(), values->end(), is_finite))
			{
				LOG_ERROR(kLogModule, "camera_matrix must contain exactly 9 finite values");
				return false;
			}

			const double fx = (*values)[0];
			const double fy = (*values)[4];

			if(fx <= 0.0 || fy <= 0.0)
			{
				LOG_ERROR(kLogModule, "camera_matrix fx (index 0) and fy (index 4) must be > 0");
				return false;
			}

			cv::Mat camera_matrix(3, 3, CV_64F);
			for(int row = 0; row < 3; ++row)
			{
				for(int col = 0; col < 3; ++col)
				{
					camera_matrix.at<double>(row, col) = (*values)[row * 3 + col];
				}
			}

			config.camera_matrix = camera_matrix;
			return true;
		}

		bool load_distort_coeffs(const toml::table& camera_table, SolverConfig& config)
		{
			const auto values =
			    read_double_array(camera_table["distort_coeffs"].as_array());

			if(!values)
			{
				LOG_ERROR(kLogModule, "distort_coeffs must be an array of numeric values");
				return false;
			}

			if(values->empty() || !std::all_of(values->begin(), values->end(), is_finite))
			{
				LOG_ERROR(kLogModule, "distort_coeffs must contain finite values");
				return false;
			}

			cv::Mat distort_coeffs(1, static_cast<int>(values->size()), CV_64F);
			for(std::size_t i = 0; i < values->size(); ++i)
			{
				distort_coeffs.at<double>(0, static_cast<int>(i)) = (*values)[i];
			}

			config.distort_coeffs = distort_coeffs;
			return true;
		}

		bool load_rotation3(const toml::table& extrinsic_table, const char* key,
		                    Eigen::Matrix3d& rotation)
		{
			const auto values = read_double_array(extrinsic_table[key].as_array());

			if(!values)
			{
				LOG_ERROR(kLogModule, "{} must be an array of numeric values", key);
				return false;
			}

			if(values->size() != 9 || !std::all_of(values->begin(), values->end(), is_finite))
			{
				LOG_ERROR(kLogModule, "{} must contain exactly 9 finite values", key);
				return false;
			}

			for(int row = 0; row < 3; ++row)
			{
				for(int col = 0; col < 3; ++col)
				{
					rotation(row, col) = (*values)[row * 3 + col];
				}
			}

			return true;
		}

		bool load_translation3(const toml::table& extrinsic_table, const char* key,
		                       Eigen::Vector3d& translation)
		{
			const auto values = read_double_array(extrinsic_table[key].as_array());

			if(!values)
			{
				LOG_ERROR(kLogModule, "{} must be an array of numeric values", key);
				return false;
			}

			if(values->size() != 3 || !std::all_of(values->begin(), values->end(), is_finite))
			{
				LOG_ERROR(kLogModule, "{} must contain exactly 3 finite values", key);
				return false;
			}

			translation = Eigen::Vector3d((*values)[0], (*values)[1], (*values)[2]);
			return true;
		}

	} // namespace

	bool load_solver_config_from_table(const toml::table& root, SolverConfig& config)
	{
		const auto* camera_table = root["camera"].as_table();
		if(camera_table == nullptr)
		{
			LOG_ERROR(kLogModule, "missing [camera] table");
			return false;
		}

		const auto* extrinsic_table = root["extrinsic"].as_table();
		if(extrinsic_table == nullptr)
		{
			LOG_ERROR(kLogModule, "missing [extrinsic] table");
			return false;
		}

		SolverConfig loaded_config;

		if(!load_camera_matrix(*camera_table, loaded_config))
		{
			return false;
		}

		if(!load_distort_coeffs(*camera_table, loaded_config))
		{
			return false;
		}

		if(!load_rotation3(*extrinsic_table, "r_camera_to_gimbal",
		                   loaded_config.r_camera_to_gimbal))
		{
			return false;
		}

		if(!load_translation3(*extrinsic_table, "t_camera_to_gimbal",
		                      loaded_config.t_camera_to_gimbal))
		{
			return false;
		}

		if(!load_rotation3(*extrinsic_table, "r_gimbal_to_imu_body",
		                   loaded_config.r_gimbal_to_imu_body))
		{
			return false;
		}

		// armor 尺寸不在 TOML 加载，保留 SolverConfig 默认值。

		config = std::move(loaded_config);

		return true;
	}

	bool load_solver_config(const std::string& config_path, SolverConfig& config)
	{
		try
		{
			const toml::table root = toml::parse_file(config_path);

			if(!load_solver_config_from_table(root, config))
			{
				LOG_ERROR(kLogModule, "solver configuration validation failed: {}", config_path);
				return false;
			}

			LOG_INFO(kLogModule, "loaded solver config: {}", config_path);

			return true;
		}
		catch(const toml::parse_error& exception)
		{
			LOG_ERROR(kLogModule, "failed to parse {}: {}", config_path, exception.what());
			return false;
		}
		catch(const std::exception& exception)
		{
			LOG_ERROR(kLogModule, "failed to load {}: {}", config_path, exception.what());
			return false;
		}
		catch(...)
		{
			LOG_ERROR(kLogModule, "failed to load {}: unknown exception", config_path);
			return false;
		}
	}

} // namespace app::auto_aim