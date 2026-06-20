add_library(tools STATIC
    crc.cpp
    logger.cpp
    maths_tools.cpp
    foxglove_comm.cpp
)

target_include_directories(tools PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/foxglove/include
    ${PROJECT_SOURCE_DIR}/tools/maths_tools/include
)

target_link_libraries(tools PUBLIC
    ${OpenCV_LIBS}
    pthread
    Eigen3::Eigen
)
