cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(SOURCE_FILES
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudautils.h
)  

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

cuda_add_library(cudaSift
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )