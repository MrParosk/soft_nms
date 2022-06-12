find_package(Torch REQUIRED)

file(GLOB SOURCES "*.cpp")
add_library(csrc SHARED ${SOURCES})
target_include_directories(csrc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(csrc ${TORCH_LIBRARIES})

install(TARGETS csrc
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
