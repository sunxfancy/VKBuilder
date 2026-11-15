add_executable(binding_validation_test main.cpp)
target_include_directories(binding_validation_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${Vulkan_INCLUDE_DIRS})
add_test(NAME binding_validation_test COMMAND binding_validation_test)

option(VKB_BUILD_REFLECTION_TESTS "Build SPIRV-Cross shader contract tests" OFF)
if(VKB_BUILD_REFLECTION_TESTS)
  find_path(VKB_SPIRV_CROSS_INCLUDE spirv_cross/spirv_cross.hpp)
  find_library(VKB_SPIRV_CROSS_CORE NAMES spirv-cross-core)
  find_program(VKB_GLSLC glslc)
  if(NOT VKB_SPIRV_CROSS_INCLUDE OR NOT VKB_SPIRV_CROSS_CORE OR NOT VKB_GLSLC)
    message(FATAL_ERROR "VKB_BUILD_REFLECTION_TESTS requires SPIRV-Cross core and glslc")
  endif()
  set(_fixtures)
  foreach(_shader resources.frag compatible.vert conflict.vert unsupported.comp row-major.vert runtime.comp nested.comp)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/${_shader}.spv")
    add_custom_command(OUTPUT "${_output}"
      COMMAND "${VKB_GLSLC}" --target-env=vulkan1.2
        "${CMAKE_CURRENT_SOURCE_DIR}/shader/${_shader}" -o "${_output}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader/${_shader}" VERBATIM)
    list(APPEND _fixtures "${_output}")
  endforeach()
  add_executable(binding_reflection_test main.cpp ${_fixtures})
  target_compile_definitions(binding_reflection_test PRIVATE VKB_TEST_REFLECTION=1)
  target_include_directories(binding_reflection_test PRIVATE ${PROJECT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS} ${VKB_SPIRV_CROSS_INCLUDE})
  target_link_libraries(binding_reflection_test PRIVATE ${VKB_SPIRV_CROSS_CORE})
  add_test(NAME binding_reflection_test COMMAND binding_reflection_test "${CMAKE_CURRENT_BINARY_DIR}")
endif()
