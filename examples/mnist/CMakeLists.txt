add_executable(mnist mnist.cpp)
set_default_target_properties(mnist)
target_link_libraries(mnist PUBLIC pico_toolshed pico_understory Eigen3::Eigen)
