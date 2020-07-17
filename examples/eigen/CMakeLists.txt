add_executable(eigen eigen.cpp)
target_link_libraries(eigen PUBLIC pico_common Eigen3::Eigen)
set_target_properties(eigen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
