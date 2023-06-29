add_executable(eigen eigen.cpp)
set_default_target_properties(eigen)
target_link_libraries(eigen PUBLIC pico_toolshed Eigen3::Eigen)
