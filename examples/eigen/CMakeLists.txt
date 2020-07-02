add_executable(eigen eigen.cpp)
target_link_libraries(eigen PUBLIC pico_common Eigen3::Eigen)
