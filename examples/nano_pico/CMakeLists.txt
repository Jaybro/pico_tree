add_executable(nano_pico)
target_sources(nano_pico
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/open_stream.cpp
        ${CMAKE_CURRENT_LIST_DIR}/nano_pico.cpp
)
target_link_libraries(nano_pico PUBLIC pico_common nanoflann::nanoflann)
set_target_properties(nano_pico
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# Only a minimum level is set for pico_tree. For this executable we want it to be 11.
set_target_properties(nano_pico PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_executable(uosr_to_bin)
target_sources(uosr_to_bin
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/format_uosr.cpp
        ${CMAKE_CURRENT_LIST_DIR}/open_stream.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uosr_to_bin.cpp
)
target_link_libraries(uosr_to_bin PUBLIC pico_common)
# This executable requires <filesystem>
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
set_target_properties(uosr_to_bin
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
