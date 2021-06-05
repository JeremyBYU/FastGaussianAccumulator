
# Set the Source, Header Files
set(HCLUST_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCLUST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCLUST_SOURCE_FILES ${HCLUST_SRC}/fastcluster.cpp)
set(HCLUST_HEADER_FILES ${HCLUST_INCLUDE_DIR}/fastcluster.h)

# Create the CPP Library
set(LIB_TYPE STATIC)
add_library(hclust-cpp ${LIB_TYPE} ${HCLUST_SOURCE_FILES})

# To include the GAConfig.hpp file
target_include_directories(hclust-cpp PRIVATE ${HCLUST_INCLUDE_DIR})
set_target_properties(hclust-cpp PROPERTIES PUBLIC_HEADER "${HCLUST_HEADER_FILES}" POSITION_INDEPENDENT_CODE ON)


# TODO - This is something that I still do not fully understand, Build vs Install?
target_include_directories(hclust-cpp PUBLIC
    $<BUILD_INTERFACE:${HCLUST_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${HCLUST_SRC})