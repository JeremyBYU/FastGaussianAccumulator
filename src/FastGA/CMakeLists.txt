# Configure Header File
configure_file("GAConfig.hpp.in" "GAConfig.hpp" @ONLY)

# Set the Source, Header Files
set(FASTGA_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(FASTGA_INCLUDE_DIR "${INCLUDE_DIR}/FastGA")
set(FASTGA_SOURCE_FILES ${FASTGA_SRC}/FastGA.cpp ${FASTGA_SRC}/GAConfig.cpp)
set(FASTGA_HEADER_FILES ${FASTGA_INCLUDE_DIR}/FastGA.hpp)

# Create the CPP Library
set(LIB_TYPE STATIC)
add_library(${FASTGA_PROJECT_NAME} ${LIB_TYPE} ${FASTGA_SOURCE_FILES})
add_library(FastGA::GA ALIAS ${FASTGA_PROJECT_NAME})

############# Start Build Options ###################

# Configure optimization
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPTIMIZATION_FLAGS "-O0 -DDEBUG")
    message("-- Configuring debug build")
else()
    set(OPTIMIZATION_FLAGS "-O3 -DNDEBUG")
    message("-- Configuring release build")
endif()

# Enable warnings, will set to compile with src/CMakeLists.txt
set(DESIRED_WARNINGS "-Wall -Wextra -Wconversion -Wunreachable-code -Wuninitialized -pedantic-errors -Wold-style-cast -Wno-error=unused-variable -Wshadow -Wfloat-equal -Weffc++")
set(MINIMAL_DESIRED_WARNINGS "-Wall -Wextra -pedantic-errors")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(DESIRED_WARNINGS "${DESIRED_WARNINGS} -Wmost")
endif()
separate_arguments(DESIRED_WARNINGS)
separate_arguments(MINIMAL_DESIRED_WARNINGS)

if (GA_BUILD_WERROR)
    set(DESIRED_WARNINGS "${DESIRED_WARNINGS} -Werror")
    set(MINIMAL_DESIRED_WARNINGS "${MINIMAL_DESIRED_WARNINGS} -Werror")
endif()

# Set GLOBAL CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}")

############# End Build Options #####################

# Set Compile Options for ONLY this target.
target_compile_options(${FASTGA_PROJECT_NAME}  PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
${DESIRED_WARNINGS}>
$<$<CXX_COMPILER_ID:MSVC>:
/W4>)

# Method to hide warnings from nanoflann using SYSTEM
get_target_property(nanoflann_include_dirs nanoflann::nanoflann INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${FASTGA_PROJECT_NAME} SYSTEM PUBLIC ${nanoflann_include_dirs})
target_link_libraries(${FASTGA_PROJECT_NAME} PRIVATE hclust-cpp)


# To include the GAConfig.hpp file
target_include_directories(${FASTGA_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# OPTION Build with OpenMP Support
if (GA_WITH_OPENMP)
  find_package(OpenMP)
  target_link_libraries(${FASTGA_PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set Properties
set_target_properties(${FASTGA_PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${FASTGA_HEADER_FILES}")
set_target_properties(${FASTGA_PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} POSITION_INDEPENDENT_CODE ON)
# Set Public and Private Include Directories
target_include_directories(${FASTGA_PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
# TODO - This is something that I still do not fully understand, Build vs Install?
target_include_directories(${FASTGA_PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${INCLUDE_DIR}>
$<INSTALL_INTERFACE:include>
PRIVATE src)
