cmake_minimum_required(VERSION 3.1)

project(cis565_sfm)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

# Crucial magic for CUDA linking
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA 10 REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

include_directories(.)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
add_subdirectory(SfM)
add_subdirectory(cudaSift)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/main.cpp"
	"src/testing_helpers.hpp"
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    SfM
    cudaSift
    ${CORELIBS}
    cublas
    curand
    cusolver
    ${OpenCV_LIBS}
    )
