add_executable(kd_forest kd_forest.cpp)
set_default_target_properties(kd_forest)
target_link_libraries(kd_forest PUBLIC pico_toolshed pico_understory)
