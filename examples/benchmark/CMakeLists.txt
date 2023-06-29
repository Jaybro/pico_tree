function(add_benchmark TARGET_NAME)
    add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
    set_default_target_properties(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            pico_toolshed
            benchmark::benchmark
    )
endfunction()

################################################################################
# bm_pico_kd_tree, bm_pico_cover_tree, bm_nanoflann, bm_opencv_flann
################################################################################

add_benchmark(bm_pico_kd_tree)

add_benchmark(bm_pico_cover_tree)
target_link_libraries(bm_pico_cover_tree PRIVATE pico_understory)

find_package(nanoflann QUIET)
if(nanoflann_FOUND)
    message(STATUS "nanoflann found. Building nanoflann benchmark.")
    add_benchmark(bm_nanoflann)
    target_link_libraries(bm_nanoflann PRIVATE nanoflann::nanoflann)
else()
    message(STATUS "nanoflann not found. nanoflann benchmark skipped.")
endif()

# The FLANN respository does not provide a flannConfig.cmake. So it's more
# easy to go with the OpenCV one.
find_package(OpenCV COMPONENTS core flann QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Building OpenCV FLANN benchmark.")
    add_benchmark(bm_opencv_flann)
    target_link_libraries(bm_opencv_flann PRIVATE ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV not found. OpenCV FLANN benchmark skipped.")
endif()

################################################################################
# uosr_to_bin
################################################################################

add_executable(uosr_to_bin uosr_to_bin.cpp)
set_default_target_properties(uosr_to_bin)
target_link_libraries(uosr_to_bin PRIVATE pico_toolshed)

################################################################################
# bin_to_ascii
################################################################################

add_executable(bin_to_ascii bin_to_ascii.cpp)
set_default_target_properties(bin_to_ascii)
target_link_libraries(bin_to_ascii PRIVATE pico_toolshed)

################################################################################
# plot_benchmarks
################################################################################

# TODO These don't get deleted when running: make clean
add_custom_target(plot_benchmarks ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/plot_benchmarks.py ${CMAKE_BINARY_DIR}/bin)
