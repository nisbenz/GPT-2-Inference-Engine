cmake_minimum_required(VERSION 3.16)
project(gpt2_inference LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# GGML directory (parent of this project when cloned, overridable via -DGGML_DIR=/path/to/ggml)
if(NOT DEFINED GGML_DIR)
    set(GGML_DIR "${CMAKE_SOURCE_DIR}/../ggml")
endif()

# GGML build directory
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Include directories
include_directories(
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-cuda
    ${CMAKE_SOURCE_DIR}/src
)

# Find GGML libraries at their actual build locations
set(GGML_LIBRARY "${GGML_BUILD_DIR}/src/libggml.so")
set(GGML_CUDA_LIBRARY "${GGML_BUILD_DIR}/src/ggml-cuda/libggml-cuda.so")

if(NOT EXISTS "${GGML_LIBRARY}")
    message(FATAL_ERROR "GGML library not found: ${GGML_LIBRARY}")
endif()
if(NOT EXISTS "${GGML_CUDA_LIBRARY}")
    message(FATAL_ERROR "GGML CUDA library not found: ${GGML_CUDA_LIBRARY}")
endif()

message(STATUS "Found GGML: ${GGML_LIBRARY}")
message(STATUS "Found GGML_CUDA: ${GGML_CUDA_LIBRARY}")

# Source files
file(GLOB SOURCES
    src/*.cpp
)

# Executable
add_executable(gpt2 ${SOURCES})

# Link GGML libraries (order matters: ggml-cuda depends on ggml)
target_link_libraries(gpt2
    ${GGML_CUDA_LIBRARY}
    ${GGML_LIBRARY}
    ${CUDA_LIBRARIES}
    pthread
    m
)

# CUDA architecture for T4 (sm_75)
set_target_properties(gpt2 PROPERTIES
    CUDA_ARCHITECTURES 75
)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
