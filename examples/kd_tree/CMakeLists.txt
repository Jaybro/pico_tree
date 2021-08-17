add_executable(kd_tree_point_traits kd_tree_point_traits.cpp)
target_link_libraries(kd_tree_point_traits PUBLIC PicoTree::PicoTree)
set_target_properties(kd_tree_point_traits
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(kd_tree kd_tree.cpp)
target_link_libraries(kd_tree PUBLIC pico_toolshed)
set_target_properties(kd_tree
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
