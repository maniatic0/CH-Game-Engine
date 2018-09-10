include(ExternalProject)

cmake_minimum_required (VERSION 3.7.2)

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


# Config
project (CH_ENGINE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/static)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/install)

# External Libraries
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

# IMPORTANT: VS doesn't care about this
# See: https://stackoverflow.com/questions/19024259/how-to-change-the-build-type-to-release-mode-in-cmake?rq=1
message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID} CONFIG=${CONFIG}")
set(MY_TEST_MODE $<$<CONFIG:Release>:Release_MODE>) #$<$<CONFIG:Debug>:DEBUG_MODE>/$<$<C_COMPILER_ID:Clang>:Clang>
message("MY_TEST_MODE=${MY_TEST_MODE} $<TARGET_FILE_DIR:myBinary>/myTest.txt")


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
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/static
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
      	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
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
    DEPENDEES glm-lib
)

set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${GLM_INCLUDE_LOCATION}"
)

include_directories(${GLM_INCLUDE_LOCATION})

# Vulkan
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})




add_executable(CH_ENGINE src/main.cpp)
add_dependencies(CH_ENGINE glfw-lib)
add_dependencies(CH_ENGINE glm-lib)
# See: https://stackoverflow.com/questions/2209929/linking-different-libraries-for-debug-and-release-builds-in-cmake-on-windows
# See: https://stackoverflow.com/questions/43456982/cmake-what-is-the-difference-between-include-directories-versus-target-link
target_link_libraries(CH_ENGINE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}glfw3dll${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(CH_ENGINE ${Vulkan_LIBRARIES})

install(TARGETS CH_ENGINE
            RUNTIME DESTINATION ${INSTALL_DIRECTORY}/bin
            LIBRARY DESTINATION ${INSTALL_DIRECTORY}/lib
            ARCHIVE DESTINATION ${INSTALL_DIRECTORY}/lib/static)

install(FILES 
  $<TARGET_FILE_DIR:CH_ENGINE>/${CMAKE_SHARED_MODULE_PREFIX}glfw3${CMAKE_SHARED_LIBRARY_SUFFIX}
  DESTINATION ${INSTALL_DIRECTORY}/bin)
