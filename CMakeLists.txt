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
    ${GGML_BUILD_DIR}/src/ggml-cpu
    ${CMAKE_SOURCE_DIR}/src
)

# Libraries: In GGML 0.10+, the ggml library contains backend registration and device enumeration
# The libggml.so links to ggml-base and backend libraries
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Check which libraries actually exist and collect full paths
# IMPORTANT: Order matters! Dependencies must come BEFORE dependents
set(GGML_LIBRARIES "")
if(EXISTS "${GGML_BUILD_DIR}/src/libggml-base.so")
    list(APPEND GGML_LIBRARIES "${GGML_BUILD_DIR}/src/libggml-base.so")
    message(STATUS "Using GGML BASE: ${GGML_BUILD_DIR}/src/libggml-base.so")
endif()
if(EXISTS "${GGML_BUILD_DIR}/src/libggml.so")
    list(APPEND GGML_LIBRARIES "${GGML_BUILD_DIR}/src/libggml.so")
    message(STATUS "Using GGML: ${GGML_BUILD_DIR}/src/libggml.so")
endif()
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-cpu/libggml-cpu.so")
    list(APPEND GGML_LIBRARIES "${GGML_BUILD_DIR}/src/ggml-cpu/libggml-cpu.so")
    message(STATUS "Using GGML CPU: ${GGML_BUILD_DIR}/src/ggml-cpu/libggml-cpu.so")
endif()
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-cuda/libggml-cuda.so")
    list(APPEND GGML_LIBRARIES "${GGML_BUILD_DIR}/src/ggml-cuda/libggml-cuda.so")
    message(STATUS "Using GGML CUDA: ${GGML_BUILD_DIR}/src/ggml-cuda/libggml-cuda.so")
endif()

if(NOT GGML_LIBRARIES)
    message(FATAL_ERROR "No GGML libraries found in ${GGML_BUILD_DIR}/src/")
endif()

# Source files
file(GLOB SOURCES
    src/*.cpp
)

# Executable
add_executable(gpt2 ${SOURCES})

# Link libraries
target_link_directories(gpt2 PRIVATE ${GGML_BUILD_DIR}/src ${GGML_BUILD_DIR}/src/ggml-cpu ${GGML_BUILD_DIR}/src/ggml-cuda)
target_link_libraries(gpt2
    "-Wl,--no-as-needed"
    ${GGML_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread
    dl
    m
)

# CUDA architecture for T4 (sm_75)
set_target_properties(gpt2 PROPERTIES
    CUDA_ARCHITECTURES 75
)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build tests
add_subdirectory(tests)