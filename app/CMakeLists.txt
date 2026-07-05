add_library(app STATIC
    auto_aim/armor.cpp
    auto_aim/armor_config.cpp
    auto_aim/solver.cpp
    auto_aim/auto_aim.cpp
)

target_include_directories(app PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_compile_definitions(app PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(app PUBLIC
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
