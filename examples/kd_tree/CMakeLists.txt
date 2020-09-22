add_executable(kd_tree kd_tree.cpp)
target_link_libraries(kd_tree PUBLIC pico_common)
set_target_properties(kd_tree
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
