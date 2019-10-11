cmake_minimum_required(VERSION 3.1)

project(cis565_structure_from_motion)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

# Crucial magic for CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA 10 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

include_directories(${GLM_INCLUDE_DIRS})
include_directories(.)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_subdirectory(SfM)
add_subdirectory(cudaSift)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/main.cpp"
    "src/glslUtility.cpp"
    "src/glslUtility.hpp"
    "src/utilityCore.hpp" 
    "src/utilityCore.cpp"
    "src/cudaMat4.hpp"
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    SfM
    cudaSift
    ${CORELIBS}
    cusolver
    cublas
    curand
    ${OpenCV_LIBS}
    ${LIBRARIES}
    )
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )