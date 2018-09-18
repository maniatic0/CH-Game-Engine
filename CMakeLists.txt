include(ExternalProject)

cmake_minimum_required (VERSION 3.9.6)

if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
    message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()


# Detect 64bit Build
# See: https://stackoverflow.com/questions/39258250/how-to-detect-if-64-bit-msvc-with-cmake
set(USE_64 ON)
if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)" 
	AND "${CMAKE_SIZEOF_VOID_P}" STREQUAL "4" 
	AND NOT CMAKE_CL_64)
    set(USE_64 OFF)
endif()


# Detect Multi Config Generator 
# See: https://gitlab.kitware.com/cmake/cmake/issues/17645
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (isMultiConfig)
    set(propagate_build_type --config $<CONFIG> -DCMAKE_BUILD_TYPE=$<CONFIG>)
else()
    set(propagate_build_type -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()


# Config
project (CH_ENGINE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/static)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/install)

# See: http://www.brianlheim.com/2018/04/09/cmake-cheat-sheet.html
set(CMAKE_CXX_STANDARD 17)

# External Libraries
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

# IMPORTANT: VS doesn't care about this
# See: https://stackoverflow.com/questions/19024259/how-to-change-the-build-type-to-release-mode-in-cmake?rq=1
message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID} CONFIG=${CONFIG}")


# GLFW
set(GLFW_LIB_NAME lib64)
set(GLFW_LIB_SUFFIX 64)
if(NOT USE_64)
   set(GLFW_LIB_NAME lib)
   set(GLFW_LIB_SUFFIX "")
endif()

set(GLFW_LOCATION ${EXTERNAL_INSTALL_LOCATION}/GLFW)

set(GLFW_INCLUDE_LOCATION ${GLFW_LOCATION}/include)
set(GLFW_LIB_LOCATION ${GLFW_LOCATION}/${GLFW_LIB_NAME})
set(GLFW_BIN_LOCATION ${GLFW_LOCATION}/bin)


# See> https://cmake.org/cmake/help/v3.0/module/ExternalProject.html
# See: https://stackoverflow.com/questions/38006584/how-to-clone-and-integrate-external-from-git-cmake-project-into-local-one
# See: https://cmake.org/cmake/help/v3.0/module/ExternalProject.html
# See: https://stackoverflow.com/questions/16842218/how-to-use-cmake-externalproject-add-or-alternatives-in-a-cross-platform-way
# See: https://stackoverflow.com/questions/25520117/cmake-install-project-doesnt-copy-my-executable-to-the-folder-i-specify
ExternalProject_Add (
	  glfw-lib
	  PREFIX ${GLFW_LOCATION}
      GIT_REPOSITORY "https://github.com/glfw/glfw.git"
      GIT_TAG "master"
      CMAKE_GENERATOR ${CMAKE_GENERATOR}
      CMAKE_GENERATOR_TOOLSET  ${CMAKE_GENERATOR_TOOLSET}
      INSTALL_COMMAND # See https://stackoverflow.com/questions/16842218/how-to-use-cmake-externalproject-add-or-alternatives-in-a-cross-platform-way
        ${CMAKE_COMMAND}
        --build .
        --target install
      CMAKE_ARGS 
    	  -DCMAKE_INSTALL_PREFIX=${GLFW_LOCATION}
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
      	#-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        ${propagate_build_type}
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      	-DLIB_SUFFIX=${GLFW_LIB_SUFFIX} 
      	-DBUILD_SHARED_LIBS=ON 
      	-DGLFW_BUILD_TESTS=OFF 
      	-DGLFW_BUILD_EXAMPLES=OFF 
      	-DGLFW_BUILD_DOCS=ON
      )

include_directories(${GLFW_INCLUDE_LOCATION})
link_directories($<TARGET_LINKER_FILE_DIR:CH_ENGINE>)

# GLM 
set(GLM_LOCATION ${EXTERNAL_INSTALL_LOCATION}/GLM)
set(GLM_INCLUDE_PRE_LOCATION ${GLM_LOCATION}/src/glm-lib/glm)
set(GLM_INCLUDE_LOCATION ${GLM_LOCATION}/include)
ExternalProject_Add (
	  glm-lib
	  PREFIX ${GLM_LOCATION}
      GIT_REPOSITORY "https://github.com/g-truc/glm.git"
      GIT_TAG "master"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${CMAKE_GENERATOR}
      CMAKE_GENERATOR_TOOLSET  ${CMAKE_GENERATOR_TOOLSET}
      CMAKE_ARGS 
    	-DCMAKE_INSTALL_PREFIX=${GLM_LOCATION}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      )

# See: https://stackoverflow.com/questions/41384447/clean-copied-files-from-external-project
# See: https://cmake.org/cmake/help/v3.0/module/ExternalProject.html
ExternalProject_Add_Step(
    glm-lib glm-copy2include
    COMMENT "Copying GLM to include folder"
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${GLM_INCLUDE_PRE_LOCATION} ${GLM_INCLUDE_LOCATION}/glm
    # See: https://stackoverflow.com/questions/41089289/externalproject-add-step-error-add-custom-command-given-append-option-with-outpu
    DEPENDEES download
)

set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${GLM_INCLUDE_LOCATION}"
)

include_directories(${GLM_INCLUDE_LOCATION})

# Vulkan
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})


# C and C++ files

file(GLOB_RECURSE CH_ENGINE_SRC
    "src/*.c"
    "src/*.cpp"
)

include_directories(include/)
add_executable(CH_ENGINE ${CH_ENGINE_SRC})
add_dependencies(CH_ENGINE glfw-lib)
add_dependencies(CH_ENGINE glm-lib)
# See: https://stackoverflow.com/questions/2209929/linking-different-libraries-for-debug-and-release-builds-in-cmake-on-windows
# See: https://stackoverflow.com/questions/43456982/cmake-what-is-the-difference-between-include-directories-versus-target-link
target_link_libraries(CH_ENGINE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}$<$<BOOL:isMultiConfig>:/$<CONFIG>>/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3dll${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(CH_ENGINE ${Vulkan_LIBRARIES})


# From vhls : https://gist.github.com/vlsh/a0d191701cb48f157b05be7f74d79396
# Added some little things for cross platform compatibility and optimizations
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator${CMAKE_EXECUTABLE_SUFFIX}")
    set(GLSL_OPTIMIZER "$ENV{VULKAN_SDK}/Bin/spirv-opt${CMAKE_EXECUTABLE_SUFFIX}")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator${CMAKE_EXECUTABLE_SUFFIX}")
    set(GLSL_OPTIMIZER "$ENV{VULKAN_SDK}/Bin32/spirv-opt${CMAKE_EXECUTABLE_SUFFIX}")
endif()

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "shaders/*.frag"
    "shaders/*.vert"
    "shaders/*.tesc"
    "shaders/*.tese"
    "shaders/*.geom"
    "shaders/*.comp"
    "shaders/*.glsl"
    "shaders/*.hlsl"
    )

set(SPIRV_BIN_FOLDER ${PROJECT_BINARY_DIR}/shaders)
set(SPIRV_OPT_PERF_FOLDER ${SPIRV_BIN_FOLDER}/performance)
set(SPIRV_OPT_SIZE_FOLDER ${SPIRV_BIN_FOLDER}/size)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${SPIRV_BIN_FOLDER}/${FILE_NAME}.spv")
    set(SPIRV_OPT_PERF "${SPIRV_OPT_PERF_FOLDER}/${FILE_NAME}.spv")
    set(SPIRV_OPT_SIZE "${SPIRV_OPT_SIZE_FOLDER}/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV} ${SPIRV_OPT_PERF} ${SPIRV_OPT_SIZE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_BIN_FOLDER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OPT_PERF_FOLDER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OPT_SIZE_FOLDER}"
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        COMMAND ${GLSL_OPTIMIZER} -O ${SPIRV} -o ${SPIRV_OPT_PERF} 
        COMMAND ${GLSL_OPTIMIZER} -Os ${SPIRV} -o ${SPIRV_OPT_SIZE}
        DEPENDS ${GLSL})

  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
  list(APPEND SPIRV_PERF_BINARY_FILES ${SPIRV_OPT_PERF})
  list(APPEND SPIRV_SIZE_BINARY_FILES ${SPIRV_OPT_SIZE})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES} ${SPIRV_PERF_BINARY_FILES} ${SPIRV_SIZE_BINARY_FILES}
    )

add_dependencies(CH_ENGINE Shaders)

add_custom_command(TARGET CH_ENGINE POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:CH_ENGINE>/shaders/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_BINARY_DIR}/shaders"
        "$<TARGET_FILE_DIR:CH_ENGINE>/shaders"
        )



install(TARGETS CH_ENGINE
            RUNTIME DESTINATION ${INSTALL_DIRECTORY}/bin
            LIBRARY DESTINATION ${INSTALL_DIRECTORY}/lib
            ARCHIVE DESTINATION ${INSTALL_DIRECTORY}/lib/static)

install(FILES 
    $<TARGET_FILE_DIR:CH_ENGINE>/${CMAKE_SHARED_MODULE_PREFIX}glfw3${CMAKE_SHARED_LIBRARY_SUFFIX}
    DESTINATION ${INSTALL_DIRECTORY}/bin)

install(DIRECTORY 
    ${PROJECT_BINARY_DIR}/shaders
    DESTINATION ${INSTALL_DIRECTORY}/bin)
