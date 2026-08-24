add_library(app STATIC
    auto_aim/armor.cpp
    auto_aim/armor_config.cpp
    auto_aim/association.cpp
    auto_aim/auto_aim.cpp
    auto_aim/solver.cpp
    auto_aim/solver_config.cpp
    auto_aim/target.cpp
    auto_aim/tracker.cpp
    auto_aim/tracker_config.cpp

    auto_aim/detector/detector.cpp
    auto_aim/detector/detector_config.cpp
    auto_aim/detector/openvino_inference.cpp
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
    openvino::runtime
)
