add_subdirectory(hclust-cpp)

include(FetchContent)
# Put any LIBRARY dependencies in here
FetchContent_Declare(
  extern_pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.4.3
)

FetchContent_Declare(
  extern_nanoflann
  GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
  GIT_TAG        2ddd9afa0de0c6a25c966ade473ed5715a331f74
)

# Add nanoflann, but dont build examples and unit tests
FetchContent_GetProperties(extern_nanoflann)
if(NOT extern_nanoflann_POPULATED)
  FetchContent_Populate(extern_nanoflann)
  set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
  add_subdirectory(${extern_nanoflann_SOURCE_DIR} ${extern_nanoflann_BINARY_DIR})
endif()


# After the following call, the library will be installed
FetchContent_MakeAvailable(extern_pybind11)