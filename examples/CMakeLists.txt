add_subdirectory(pico_common)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    add_subdirectory(eigen)
endif()

add_subdirectory(range_query)

find_package(nanoflann)
if(nanoflann_FOUND)
    add_subdirectory(nano_pico)
endif()
