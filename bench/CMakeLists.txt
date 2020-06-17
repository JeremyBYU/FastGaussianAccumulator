include(FetchContent)

FetchContent_Declare(
  extern_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.5.0
)

# This module depend on having googlebenchmark installed
FetchContent_GetProperties(extern_googlebenchmark)
if(NOT extern_googlebenchmark_POPULATED)
  FetchContent_Populate(extern_googlebenchmark)
  SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
  add_subdirectory(${extern_googlebenchmark_SOURCE_DIR} ${extern_googlebenchmark_BINARY_DIR})
endif()

# Google Benchmark needs threads!
find_package(Threads REQUIRED)

set(BENCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/run-bench.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/hilbert_bench.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/kd_bench.cpp")
set(BENCH_LINK_LIBRARIES benchmark ${PROJECT_NAME} Threads::Threads)

if(${GA_WITH_S2_GEOMETRY})
    list(APPEND BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/run-bench.cpp)
    list(APPEND BENCH_LINK_LIBRARIES s2)
endif()

set(npy ${PROJECT_SOURCE_DIR}/thirdparty)
add_executable(run-bench ${BENCH_SOURCES})
target_link_libraries(run-bench PRIVATE ${BENCH_LINK_LIBRARIES})
target_include_directories(run-bench PRIVATE ${npy})
