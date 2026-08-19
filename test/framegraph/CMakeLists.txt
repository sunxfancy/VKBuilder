add_executable(framegraph_test main.cpp)
target_include_directories(framegraph_test
    PRIVATE ${CMAKE_SOURCE_DIR}/include ${Vulkan_INCLUDE_DIRS})
target_link_libraries(framegraph_test ${Vulkan_LIBRARY})
add_test(NAME framegraph_test COMMAND framegraph_test)
