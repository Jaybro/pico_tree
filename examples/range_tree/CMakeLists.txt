add_executable(range_tree range_tree.cpp)
target_link_libraries(range_tree PUBLIC pico_common)
set_target_properties(range_tree
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
