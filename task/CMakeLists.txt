# test
# 分组注册 CTest：快速、稳定、无硬件/真实视频依赖的单元测试。
# test_auto_aim_pipeline / test_all_pipeline（真实 OpenVINO + demo.avi）
# 保持手动 integration/replay test，不注册进默认 CTest。
add_test(NAME test_solver COMMAND test_solver)
add_test(NAME test_detector COMMAND test_detector)
add_test(NAME test_openvino_inference COMMAND test_openvino_inference)
add_test(NAME test_auto_aim COMMAND test_auto_aim)
add_test(NAME test_armor COMMAND test_armor)
add_test(NAME test_solver_config COMMAND test_solver_config)
add_test(NAME test_extended_kalman_filter COMMAND test_extended_kalman_filter)
add_test(NAME test_target COMMAND test_target)
add_test(NAME test_association COMMAND test_association)
add_test(NAME test_tracker COMMAND test_tracker)
add_test(NAME test_tracker_config COMMAND test_tracker_config)
add_test(NAME test_tracker_spherical_equivalence COMMAND test_tracker_spherical_equivalence)
add_test(NAME test_trajectory COMMAND test_trajectory)
add_test(NAME test_vehicle_prediction COMMAND test_vehicle_prediction)
add_test(NAME test_aimer_config COMMAND test_aimer_config)
add_test(NAME test_aimer COMMAND test_aimer)
add_test(NAME test_aimer_pipeline COMMAND test_aimer_pipeline)

add_executable(sentry sentry.cpp)
add_executable(read_toml test/read_toml.cpp)
add_executable(test_serial test/test_serial.cpp)
add_executable(test_logger test/test_logger.cpp)
add_executable(test_foxglove test/test_foxglove.cpp)
add_executable(test_camera test/test_camera.cpp)
add_executable(test_armor test/test_armor.cpp)
add_executable(test_solver test/test_solver.cpp)
add_executable(test_usb_camera test/test_usb_camera.cpp)
add_executable(test_detector test/test_detector.cpp)
add_executable(test_openvino_inference test/test_openvino_inference.cpp)
add_executable(test_auto_aim test/test_auto_aim.cpp)
add_executable(test_auto_aim_visual test/test_auto_aim_visual.cpp)
add_executable(test_auto_aim_pipeline test/test_auto_aim_pipeline.cpp)
add_executable(test_solver_config test/test_solver_config.cpp)
add_executable(test_extended_kalman_filter test/test_extended_kalman_filter.cpp)
add_executable(test_target test/test_target.cpp)
add_executable(test_association test/test_association.cpp)
add_executable(test_tracker test/test_tracker.cpp)
add_executable(test_tracker_config test/test_tracker_config.cpp)
add_executable(test_tracker_spherical_equivalence test/test_tracker_spherical_equivalence.cpp)
add_executable(test_tracker_replay test/test_tracker_replay.cpp)
add_executable(test_tracker_visual test/test_tracker_visual.cpp)
add_executable(test_all_pipeline test/test_all_pipeline.cpp)
add_executable(test_trajectory test/test_trajectory.cpp)
add_executable(test_vehicle_prediction test/test_vehicle_prediction.cpp)
add_executable(test_aimer_config test/test_aimer_config.cpp)
add_executable(test_aimer test/test_aimer.cpp)
add_executable(test_aimer_pipeline test/test_aimer_pipeline.cpp)

target_compile_definitions(test_logger PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_definitions(sentry PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_serial PRIVATE io tools)

target_link_libraries(sentry PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_logger PRIVATE tools)

target_link_libraries(test_trajectory PRIVATE tools)

target_link_libraries(test_vehicle_prediction PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_aimer_config PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_aimer PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_aimer_pipeline PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_foxglove PRIVATE tools)

target_link_libraries(test_camera PRIVATE ${ARAVIS_LIBRARIES} ${OpenCV_LIBS})

target_include_directories(test_detector PRIVATE
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(test_detector PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_include_directories(test_solver PRIVATE
    ${PROJECT_SOURCE_DIR}
)

target_compile_definitions(test_armor PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
target_compile_definitions(test_solver PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_armor PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_compile_definitions(test_openvino_inference PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_openvino_inference PRIVATE
    app
    tools
    ${OpenCV_LIBS}
)

target_include_directories(test_auto_aim PRIVATE
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(test_auto_aim PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_compile_definitions(test_auto_aim_visual PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_definitions(test_auto_aim_pipeline PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_auto_aim_visual PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_auto_aim_pipeline PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_include_directories(test_solver_config PRIVATE
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(test_solver_config PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_solver PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_extended_kalman_filter PRIVATE
    tools
    Eigen3::Eigen
)

target_link_libraries(test_target PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_association PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_tracker PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_tracker_config PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_tracker_spherical_equivalence PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_compile_definitions(test_tracker_visual PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_definitions(test_tracker_replay PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_definitions(test_all_pipeline PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_tracker_replay PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_tracker_visual PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_all_pipeline PRIVATE
    app
    tools
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_compile_definitions(test_usb_camera PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(test_usb_camera
    PRIVATE
    io
    tools
    ${OpenCV_LIBS}
)