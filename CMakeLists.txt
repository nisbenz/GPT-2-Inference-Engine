cmake_minimum_required(VERSION 3.16)
project(gpt2_inference LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GGML directory (parent of this project when cloned, overridable via -DGGML_DIR=/path/to/ggml)
if(NOT DEFINED GGML_DIR)
    set(GGML_DIR "${CMAKE_SOURCE_DIR}/../ggml")
endif()

# Include directories
include_directories(
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${CMAKE_SOURCE_DIR}/src
)

# Find GGML libraries (GGML builds libggml.so and libggml_cublas.so)
set(GGML_BUILD_DIR "${GGML_DIR}/build")

find_library(GGML_LIBRARY ggml PATHS ${GGML_BUILD_DIR}/lib NO_DEFAULT_PATH)
find_library(GGML_CUBLAS_LIBRARY ggml_cublas PATHS ${GGML_BUILD_DIR}/lib NO_DEFAULT_PATH)

if(NOT GGML_LIBRARY)
    message(FATAL_ERROR "GGML library not found in ${GGML_BUILD_DIR}/lib")
endif()
if(NOT GGML_CUBLAS_LIBRARY)
    message(FATAL_ERROR "GGML CUDA library not found in ${GGML_BUILD_DIR}/lib")
endif()

message(STATUS "Found GGML: ${GGML_LIBRARY}")
message(STATUS "Found GGML_CUBLAS: ${GGML_CUBLAS_LIBRARY}")

# Source files
file(GLOB SOURCES
    src/*.cpp
)

# Executable
add_executable(gpt2 ${SOURCES})

# Link GGML libraries
target_link_libraries(gpt2
    ${GGML_LIBRARY}
    ${GGML_CUBLAS_LIBRARY}
    CUDA::cublas
    pthread
    m
)

# CUDA architecture for T4 (sm_75)
set_target_properties(gpt2 PROPERTIES
    CUDA_ARCHITECTURES 75
)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
