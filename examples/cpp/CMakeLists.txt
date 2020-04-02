#### Create Examples

find_package(Threads REQUIRED)

add_executable(example-kd "${CMAKE_CURRENT_SOURCE_DIR}/example_kd.cpp")
target_link_libraries(example-kd ${PROJECT_NAME} Threads::Threads)
add_executable(example-icochart "${CMAKE_CURRENT_SOURCE_DIR}/example_icochart.cpp")
target_link_libraries(example-icochart ${PROJECT_NAME}  Threads::Threads)
add_executable(example-nano "${CMAKE_CURRENT_SOURCE_DIR}/example_nano.cpp")
target_link_libraries(example-nano ${PROJECT_NAME} Threads::Threads)

if(${GA_WITH_S2_GEOMETRY})
    add_executable(example-s2 "${CMAKE_CURRENT_SOURCE_DIR}/example_s2.cpp")
    target_link_libraries(example-s2 ${PROJECT_NAME} s2)
endif()