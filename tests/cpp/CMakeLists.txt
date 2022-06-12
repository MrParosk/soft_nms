find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

add_executable(run_tests tests.cpp)
target_link_libraries(run_tests csrc ${GTEST_LIBRARIES} ${TORCH_LIBRARIES})
