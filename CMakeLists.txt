cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(soft_nms)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(run_tests tests/cpp/tests.cpp)
target_sources(run_tests PRIVATE "${CURRENT_DIR}/pt_soft_nms/csrc/soft_nms.cpp" )
target_link_libraries(run_tests  ${GTEST_LIBRARIES} ${TORCH_LIBRARIES})
