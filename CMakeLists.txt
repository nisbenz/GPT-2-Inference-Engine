cmake_minimum_required(VERSION 3.16)
project(gpt2_inference LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GGML directory (parent of this project when cloned)
set(GGML_DIR "${CMAKE_SOURCE_DIR}/../ggml")

# Include directories
include_directories(
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${CMAKE_SOURCE_DIR}/src
)

# Find GGML libraries
find_library(GGML_LIBRARY NAMES ggml PATHS ${GGML_DIR}/build/lib NO_DEFAULT_PATH)
find_library(GGML_CUBLAS_LIBRARY NAMES ggml_cublas PATHS ${GGML_DIR}/build/lib NO_DEFAULT_PATH)

# Source files
file(GLOB SOURCES
    src/*.cpp
)

# Executable
add_executable(gpt2 ${SOURCES})

# Link libraries
target_link_libraries(gpt2
    ${GGML_LIBRARY}
    ${GGML_CUBLAS_LIBRARY}
    cublas
    pthread
    m
)

# CUDA architecture for T4 (sm_75)
set_target_properties(gpt2 PROPERTIES
    CUDA_ARCHITECTURES 75
)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
