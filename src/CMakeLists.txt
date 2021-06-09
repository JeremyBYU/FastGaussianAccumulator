############ Start Version Parsing ##################

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" FASTGA_VERSION_READ)
foreach(ver ${FASTGA_VERSION_READ})
    if (ver MATCHES "(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(FASTGA_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
string(CONCAT FASTGA_VERSION
    "${FASTGA_VERSION_MAJOR}"
    ".${FASTGA_VERSION_MINOR}"
    ".${FASTGA_VERSION_PATCH}"
)

set(PROJECT_VERSION "${FASTGA_VERSION}")

message("Version is ${PROJECT_VERSION}")

############# End Version Parsing ##################


add_subdirectory(FastGA)

if (GA_BUILD_PYMODULE)
    add_subdirectory(Python)
endif ()
