cmake_minimum_required(VERSION 3.10)
project(yolov2_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(yolov2_detect
    src/yolov2_main.cpp
    src/yolo_image.cpp
    src/yolo_post.cpp
    src/yolo_cfg.cpp
    src/yolo_math.cpp
    src/yolo_region.cpp
    src/yolo_layers.cpp
    src/yolo_net.cpp
    src/yolo_utils.cpp
    hls/yolo2_accel.cpp
    hls/core_io.cpp
    hls/core_compute.cpp
    hls/core_scheduler.cpp
    hls/model_config.cpp
    hls/yolo2_model.cpp
    src/stb_image_implementation.cpp
)

target_include_directories(yolov2_detect PRIVATE include hls)
target_compile_definitions(yolov2_detect PRIVATE REORG_TEST)
target_compile_options(yolov2_detect PRIVATE -O3 -Wall -Wextra)
target_link_libraries(yolov2_detect PRIVATE m)

# TODO: expose backend selection and HLS/GEN variants as options when the
# codebase is further modularized.
