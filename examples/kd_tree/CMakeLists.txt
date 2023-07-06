function(add_demo_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
    set_default_target_properties(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE pico_toolshed)
endfunction()

add_demo_executable(kd_tree_minimal)

add_demo_executable(kd_tree_creation)

add_demo_executable(kd_tree_search)

add_demo_executable(kd_tree_dynamic_arrays)

add_demo_executable(kd_tree_custom_point_type)

add_demo_executable(kd_tree_custom_space_type)

add_demo_executable(kd_tree_custom_search_visitor)

add_demo_executable(kd_tree_save_and_load)
