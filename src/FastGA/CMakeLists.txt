configure_file("GAConfig.hpp.in" "GAConfig.hpp" @ONLY)

# Set the Source, Header Files
set(FASTGA_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(FASTGA_INCLUDE_DIR "${INCLUDE_DIR}/FastGA")
set(FASTGA_SOURCE_FILES ${FASTGA_SRC}/FastGA.cpp ${FASTGA_SRC}/GAConfig.cpp)
set(FASTGA_HEADER_FILES ${FASTGA_INCLUDE_DIR}/FastBinarySearch.hpp ${FASTGA_INCLUDE_DIR}/Helper.hpp
${FASTGA_INCLUDE_DIR}/FastGA.hpp ${FASTGA_INCLUDE_DIR}/Ico.hpp
${FASTGA_INCLUDE_DIR}/NanoFlannAdaptor.hpp)

# Create the CPP Library
set(LIB_TYPE STATIC)
add_library(${PROJECT_NAME} ${LIB_TYPE} ${FASTGA_SOURCE_FILES})
add_library(FastGA::GA ALIAS ${PROJECT_NAME})

# To include the GAConfig.hpp file
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Method to hide warnings from nanoflann using SYSTEM
get_target_property(nanoflann_include_dirs nanoflann::nanoflann INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${nanoflann_include_dirs})

target_link_libraries(${PROJECT_NAME} PRIVATE hclust-cpp)

# Set Compile Options for ONLY this target.
target_compile_options(${PROJECT_NAME}  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    ${DESIRED_WARNINGS}>
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4>)

# OPTION Build with OpenMP Support
if (GA_WITH_OPENMP)
    find_package(OpenMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set Properties
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${FASTGA_HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} POSITION_INDEPENDENT_CODE ON)
# Set Public and Private Include Directories
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
# TODO - This is something that I still do not fully understand, Build vs Install?
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
