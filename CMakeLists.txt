cmake_minimum_required(VERSION 3.16)
project(TGU_Robocore_2027)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认采用 Debug
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Boost CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARAVIS REQUIRED aravis-0.8)

find_package(Eigen3 REQUIRED)

find_package(OpenVINO REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${ARAVIS_INCLUDE_DIRS})

# ===== 子模块 =====
add_subdirectory(io)
add_subdirectory(tools)
add_subdirectory(app)
add_subdirectory(task)


# 编译方式: cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug / cmake -S . -B build -DCMAKE_BUILD_TYPE=Release