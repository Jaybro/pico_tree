function(add_benchmark BM_TARGET_NAME)
    add_executable(${BM_TARGET_NAME})
    target_sources(${BM_TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/${BM_TARGET_NAME}.cpp
    )
    target_link_libraries(${BM_TARGET_NAME}
        PRIVATE
            pico_common
            benchmark::benchmark
    )
    set_target_properties(${BM_TARGET_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

################################################################################
# bm_nanoflann, bm_pico_kd_tree, bm_pico_cover_tree
################################################################################

add_benchmark(bm_nanoflann)
target_link_libraries(bm_nanoflann
    PRIVATE
         nanoflann::nanoflann
)
add_benchmark(bm_pico_kd_tree)
add_benchmark(bm_pico_cover_tree)

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
