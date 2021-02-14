cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(soft_nms)

find_package(Torch REQUIRED)

add_library(soft_nms SHARED pt_soft_nms/csrc/op.cpp)
target_link_libraries(soft_nms "${TORCH_LIBRARIES}")
set_property(TARGET soft_nms PROPERTY CXX_STANDARD 14)
