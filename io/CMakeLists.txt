cmake_minimum_required(VERSION 3.16)

add_library(io STATIC
    serial/serial.cpp
    hikrobot/hikrobot.cpp
    usbcamera/usbcamera.cpp
    usbcamera/usbcamera_config.cpp
)

target_link_libraries(io
    PUBLIC
    tools
    ${OpenCV_LIBS}
    boost_system
    pthread
)

target_include_directories(io
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
