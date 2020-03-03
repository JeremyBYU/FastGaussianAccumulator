cmake_minimum_required(VERSION 3.11)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
# Project Details
set(PROJECT_NAME "FastGA")
set(PROJECT_VERSION 0.0.1)
set(PROJECT_EMAIL   "")
set(PROJECT_HOME    "")
set(PROJECT_DOCS    "")
set(PROJECT_CODE    "")
set(PROJECT_ISSUES  "")
# Set Project Properties
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION}
DESCRIPTION "Fast Guassian Accumulator Library"
LANGUAGES CXX)
# Set Global Properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(VERBOSE OFF)

# Output Folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Build Options
option(WERROR "Add -Werror flag to build (turns warnings into errors)" OFF)

# Configure optimization
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
set(OPTIMIZATION_FLAGS "-O0 -DDEBUG")
message("-- Configuring debug build")
else()
set(OPTIMIZATION_FLAGS "-O3 -DNDEBUG")
message("-- Configuring release build")
endif()

# Enable extra warnings to adhere to https://github.com/mapbox/cpp/issues/37
set(DESIRED_WARNINGS "-Wall -Wextra -Wconversion -Wunreachable-code -Wuninitialized -pedantic-errors -Wold-style-cast -Wno-error=unused-variable -Wshadow -Wfloat-equal -Weffc++")
set(MINIMAL_DESIRED_WARNINGS "-Wall -Wextra -pedantic-errors")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
set(DESIRED_WARNINGS "${DESIRED_WARNINGS} -Wmost")
endif()
separate_arguments(DESIRED_WARNINGS)
separate_arguments(MINIMAL_DESIRED_WARNINGS)

### Set GLOBAL CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}")
if (WERROR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

#### Project Dependencies
find_package(OpenMP)
# NanoFlann
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
add_subdirectory("thirdparty/nanoflann")
### Google Testing and Benchmarks Libraries
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library." FORCE)
add_subdirectory("thirdparty/googletest")
add_subdirectory("thirdparty/benchmark")
add_subdirectory("thirdparty/Catch2")

set(GTEST_ROOT "${PROJECT_SOURCE_DIR}/thirdparty/googletest/googletest" CACHE STRING "" FORCE)
# set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("thirdparty/s2geometry")


### Set Project File List
set(GI_SRC_DIR "${PROJECT_SOURCE_DIR}/src/FastGA")
set(GI_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(GI_SRC_FILES ${GI_SRC_DIR}/FastGA.cpp ${GI_SRC_DIR}/Helper.cpp)
file(GLOB ALL_HEADERS ${GI_INCLUDE_DIR}/FastGA/*.hpp)
set(GI_HEADER_FILES ${GI_INCLUDE_DIR}/FastGA.hpp ALL_HEADERS)

### Create Library for Gaussian Accumulator
set(LIB_TYPE STATIC)
add_library(${PROJECT_NAME} ${LIB_TYPE} "${GI_SRC_FILES}")
add_library(FastGA::GA ALIAS ${PROJECT_NAME})

### Enable warnings for ONLY this target. Error on them
target_compile_options(${PROJECT_NAME}  PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
${DESIRED_WARNINGS}>
$<$<CXX_COMPILER_ID:MSVC>:
/W4>)

### Set Header and Properties of Library
target_include_directories(${PROJECT_NAME} PRIVATE ${GI_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${GI_INCLUDE_DIR}>
$<INSTALL_INTERFACE:include>
PRIVATE src)

get_target_property(nanoflann_include_dirs nanoflann::nanoflann INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${nanoflann_include_dirs})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${GI_HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} POSITION_INDEPENDENT_CODE ON)
# find_program(CLANG_TIDY
#   NAMES clang-tidy-7 clang-tidy-6.0 clang-tidy-5.0 clang-tidy-4.0 clang-tidy)
# set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY})

#### Testing Executables
file(GLOB TEST_SOURCES test/*.cpp)
add_executable(unit-tests ${TEST_SOURCES})
target_link_libraries(unit-tests benchmark::benchmark OpenMP::OpenMP_CXX Catch2::Catch2 ${PROJECT_NAME} )

#### Benchmark Executables
file(GLOB BENCH_SOURCES bench/*.cpp)
add_executable(bench-run ${BENCH_SOURCES})
set(BENCHMARK_INCLUDE ${PROJECT_SOURCE_DIR}/bench)
target_include_directories(bench-run PRIVATE ${BENCHMARK_INCLUDE})
target_link_libraries(bench-run benchmark::benchmark OpenMP::OpenMP_CXX ${PROJECT_NAME} s2)

#### Example Executables
add_executable(example-kd "examples/cpp/example_kd.cpp")
target_link_libraries(example-kd ${PROJECT_NAME})
add_executable(example-nano "examples/cpp/example_nano.cpp")
target_link_libraries(example-nano ${PROJECT_NAME})
add_executable(example-s2 "examples/cpp/example_s2.cpp")
target_link_libraries(example-s2 ${PROJECT_NAME} s2)

# Python Bindings
if(NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "fastga")
endif()
add_subdirectory("thirdparty/pybind11")
add_subdirectory(src/Python)

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
