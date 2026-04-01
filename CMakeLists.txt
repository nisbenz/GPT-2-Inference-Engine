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

# Search multiple possible locations for each library
macro(find_ggml_lib VAR_NAME LIB_NAME)
    set(${VAR_NAME} "")
    # Try common locations - libraries may be directly in src/ or in subdirectories
    set(_search_paths
        "${GGML_BUILD_DIR}/src/${LIB_NAME}.so"
        "${GGML_BUILD_DIR}/src/${LIB_NAME}.so.0"
        "${GGML_BUILD_DIR}/src/${LIB_NAME}/lib${LIB_NAME}.so"
        "${GGML_BUILD_DIR}/src/${LIB_NAME}/lib${LIB_NAME}.so.0"
        "${GGML_BUILD_DIR}/src/ggml-cpu/${LIB_NAME}.so"
        "${GGML_BUILD_DIR}/src/ggml-cpu/${LIB_NAME}.so.0"
        "${GGML_BUILD_DIR}/src/ggml-cuda/${LIB_NAME}.so"
        "${GGML_BUILD_DIR}/src/ggml-cuda/${LIB_NAME}.so.0"
    )
    foreach(_path ${_search_paths})
        if(EXISTS "${_path}" AND "${${VAR_NAME}}" STREQUAL "")
            set(${VAR_NAME} "${_path}")
            message(STATUS "Found ${LIB_NAME}: ${_path}")
        endif()
    endforeach()
    if("${${VAR_NAME}}" STREQUAL "")
        message(WARNING "Could not find ${LIB_NAME}")
    endif()
endmacro()

find_ggml_lib(GGML_BASE_LIB "libggml-base")
find_ggml_lib(GGML_CORE_LIB "libggml")
find_ggml_lib(GGML_CPU_LIB "libggml-cpu")
find_ggml_lib(GGML_CUDA_LIB "libggml-cuda")

# Add in dependency order (base first, then dependents)
if(NOT "${GGML_BASE_LIB}" STREQUAL "")
    list(APPEND GGML_LIBRARIES "${GGML_BASE_LIB}")
endif()
if(NOT "${GGML_CORE_LIB}" STREQUAL "")
    list(APPEND GGML_LIBRARIES "${GGML_CORE_LIB}")
endif()
if(NOT "${GGML_CPU_LIB}" STREQUAL "")
    list(APPEND GGML_LIBRARIES "${GGML_CPU_LIB}")
endif()
if(NOT "${GGML_CUDA_LIB}" STREQUAL "")
    list(APPEND GGML_LIBRARIES "${GGML_CUDA_LIB}")
endif()

if(NOT GGML_LIBRARIES)
    message(FATAL_ERROR "No GGML libraries found in ${GGML_BUILD_DIR}/src/")
endif()

message(STATUS "GGML libraries to link: ${GGML_LIBRARIES}")

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