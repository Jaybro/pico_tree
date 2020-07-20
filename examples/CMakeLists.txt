add_subdirectory(pico_common)

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    add_subdirectory(eigen)
    message(STATUS "Eigen3 found. Building Eigen example.")
else()
    message(STATUS "Eigen3 not found. Eigen example skipped.")
endif()

add_subdirectory(range_query)

find_package(nanoflann QUIET)
if(nanoflann_FOUND)
    add_subdirectory(nano_pico)
    message(STATUS "nanoflann found. Building nanoflann example.")
else()
    message(STATUS "nanoflann not found. nanoflann example skipped.")
endif()
