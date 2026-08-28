# TinyMPC（第三方 vendor，保持原样，见 tools/tinympc/README.md）。
# 只暴露 tinympc 目录，不暴露整个 PROJECT_SOURCE_DIR。
add_library(tinympc STATIC
    tinympc/admm.cpp
    tinympc/tiny_api.cpp
    tinympc/rho_benchmark.cpp
)

target_include_directories(tinympc PUBLIC
    ${PROJECT_SOURCE_DIR}/tools/tinympc
)

target_link_libraries(tinympc PUBLIC
    Eigen3::Eigen
)

add_library(tools STATIC
    crc.cpp
    extended_kalman_filter.cpp
    img_tools.cpp
    logger.cpp
    logger_config_loader.cpp
    maths_tools.cpp
    trajectory.cpp
    tiny_mpc_2d.cpp
    foxglove_comm.cpp
)

target_include_directories(tools PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/foxglove/include
    ${PROJECT_SOURCE_DIR}/tools/maths_tools/include
)

target_link_libraries(tools PUBLIC
    tinympc
    ${OpenCV_LIBS}
    pthread
    Eigen3::Eigen
)
