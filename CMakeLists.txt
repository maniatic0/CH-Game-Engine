include(ExternalProject)

cmake_minimum_required (VERSION 2.6)

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

# External Libraries
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

# GLFW
set(GLFW_LIB_NAME lib64)
set(GLFW_LIB_SUFFIX 64)
if(NOT USE_64)
   set(GLFW_LIB_NAME lib)
   set(GLFW_LIB_SUFFIX "")
endif()
message(STATUS "USE_64=${USE_64} GLFW_LIB_SUFFIX=${GLFW_LIB_SUFFIX}")

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
      UPDATE_COMMAND ""
      CMAKE_GENERATOR ${CMAKE_GENERATOR}
      CMAKE_GENERATOR_TOOLSET  ${CMAKE_GENERATOR_TOOLSET}
      INSTALL_COMMAND # See https://stackoverflow.com/questions/16842218/how-to-use-cmake-externalproject-add-or-alternatives-in-a-cross-platform-way
        ${CMAKE_COMMAND}
        --build .
        --target install
      CMAKE_ARGS 
    	-DCMAKE_INSTALL_PREFIX=${GLFW_LOCATION}
      	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      	-DLIB_SUFFIX=${GLFW_LIB_SUFFIX} 
      	-DBUILD_SHARED_LIBS=ON 
      	-DGLFW_BUILD_TESTS=OFF 
      	-DGLFW_BUILD_EXAMPLES=OFF 
      	-DGLFW_BUILD_DOCS=OFF
      )


include_directories(${GLFW_INCLUDE_LOCATION})
link_directories(${GLFW_LIB_LOCATION})


# GLM 
set(GLM_LOCATION ${EXTERNAL_INSTALL_LOCATION}/GLM)

set(GLM_INCLUDE_LOCATION ${GLM_LOCATION}/src/glm-lib/glm)
ExternalProject_Add (
	  glm-lib
	  PREFIX ${GLM_LOCATION}
      GIT_REPOSITORY "https://github.com/g-truc/glm.git"
      GIT_TAG "master"
      UPDATE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${CMAKE_GENERATOR}
      CMAKE_GENERATOR_TOOLSET  ${CMAKE_GENERATOR_TOOLSET}
      CMAKE_ARGS 
    	-DCMAKE_INSTALL_PREFIX=${GLM_LOCATION}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      )

include_directories(${GLM_INCLUDE_LOCATION})

project (CH_ENGINE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_executable(CH_ENGINE src/main.cpp)
add_dependencies(CH_ENGINE glfw-lib)
add_dependencies(CH_ENGINE glm-lib)
# See: https://stackoverflow.com/questions/2209929/linking-different-libraries-for-debug-and-release-builds-in-cmake-on-windows
# See: https://stackoverflow.com/questions/43456982/cmake-what-is-the-difference-between-include-directories-versus-target-link
target_link_libraries(CH_ENGINE ${GLFW_LIB_LOCATION}/glfw3dll.lib)	
find_package(Vulkan REQUIRED)
