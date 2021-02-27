################################################################################
# bm_nanoflann
################################################################################

add_executable(bm_nanoflann)
target_sources(bm_nanoflann
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bm_nanoflann.cpp
)
target_link_libraries(bm_nanoflann
    PRIVATE
        pico_common
        nanoflann::nanoflann
        benchmark::benchmark
)
set_target_properties(bm_nanoflann
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

################################################################################
# bm_pico_kd_tree
################################################################################

add_executable(bm_pico_kd_tree)
target_sources(bm_pico_kd_tree
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bm_pico_kd_tree.cpp
)
target_link_libraries(bm_pico_kd_tree
    PRIVATE
        pico_common
        benchmark::benchmark
)
set_target_properties(bm_pico_kd_tree
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

################################################################################
# uosr_to_bin
################################################################################

add_executable(uosr_to_bin)
target_sources(uosr_to_bin
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/format_uosr.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uosr_to_bin.cpp
)
target_link_libraries(uosr_to_bin PRIVATE pico_common)
# This executable requires <filesystem>
target_compile_features(uosr_to_bin PRIVATE cxx_std_17)
set_target_properties(uosr_to_bin
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

################################################################################
# bin_to_ascii
################################################################################

add_executable(bin_to_ascii)
target_sources(bin_to_ascii
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bin_to_ascii.cpp
)
target_link_libraries(bin_to_ascii PRIVATE pico_common)
# This executable requires <filesystem>
target_compile_features(bin_to_ascii PRIVATE cxx_std_17)
set_target_properties(bin_to_ascii
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

################################################################################
# plot_benchmarks
################################################################################

# TODO These don't get deleted when running: make clean
add_custom_target(plot_benchmarks ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/plot_benchmarks.py ${CMAKE_BINARY_DIR}/bin)
