add_executable(benchmark)
target_sources(benchmark
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/benchmark.cpp
)
target_link_libraries(benchmark
    PRIVATE
        pico_common
        nanoflann::nanoflann
        benchmark::benchmark
)
set_target_properties(benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# For the benchmark demo we want to explicitly use C++11.
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

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

file(COPY ${CMAKE_CURRENT_LIST_DIR}/plot_benchmarks.py 
    DESTINATION ${CMAKE_BINARY_DIR}/bin)
