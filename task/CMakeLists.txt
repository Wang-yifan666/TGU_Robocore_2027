# test
add_executable(read_toml test/read_toml.cpp)
add_executable(test_serial test/test_serial.cpp)
add_executable(test_logger test/test_logger.cpp)
add_executable(test_foxglove test/test_foxglove.cpp)
add_executable(test_camera test/test_camera.cpp)
add_executable(test_armor test/test_armor.cpp)

target_link_libraries(test_serial PRIVATE io tools)
target_link_libraries(test_logger PRIVATE tools)
target_link_libraries(test_foxglove PRIVATE tools)
target_link_libraries(test_camera PRIVATE ${ARAVIS_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(test_armor PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_compile_definitions(test_armor PRIVATE
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(test_armor PRIVATE app tools ${OpenCV_LIBS} Eigen3::Eigen)
