cmake_minimum_required(VERSION 3.14)
# Project Details
set(PROJECT_NAME "FastGA")
set(PROJECT_VERSION 0.0.1)
set(PROJECT_EMAIL   "")
set(PROJECT_HOME    "")
set(PROJECT_DOCS    "")
set(PROJECT_CODE    "")
set(PROJECT_ISSUES  "")
set(PYPI_PACKAGE_NAME "fastgac")
# Set Project Properties
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION}
DESCRIPTION "Fast Guassian Accumulator Library"
LANGUAGES CXX)
# Set Global Properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Include cmake folder
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Extras.cmake)

set(ORIG_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Output Folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Options for GA
option(GA_BUILD_PYMODULE "GA -Build Python Module" ON)
option(GA_BUILD_TESTS "GA - Build Tests" ON)
option(GA_BUILD_BENCHMARKS "GA - Build Benchmarks" ON)
option(GA_BUILD_EXAMPLES "GA - Build Examples" ON)
option(GA_WITH_OPENMP "GA - Build with OpenMP Support" ON)
option(GA_WITH_S2_GEOMETRY "GA - Build Google S2 Geometry for benchmark/example comparisons" OFF)
option(GA_BUILD_WERROR "GA - Add Werror flag to build (turns warnings into errors)" OFF)

# Add any dependencies needed by our library
add_subdirectory("thirdparty")

# Build our library
add_subdirectory("src")

# Build examples if configured
if(GA_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()

# Build tests if configured
if(GA_BUILD_TESTS)
    add_subdirectory("tests")
endif()

# Build benchmarks if configured
if(GA_BUILD_BENCHMARKS)
    add_subdirectory("bench")
endif()


