cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ts_ops)

find_package(Torch REQUIRED)

# add_executable(ts_ops soft_nms.cpp)
add_library(ts_ops SHARED soft_nms.cpp)
target_link_libraries(ts_ops "${TORCH_LIBRARIES}")
set_property(TARGET ts_ops PROPERTY CXX_STANDARD 14)
