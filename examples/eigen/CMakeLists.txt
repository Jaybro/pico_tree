add_executable(eigen eigen.cpp)
target_link_libraries(eigen PUBLIC pico_common Eigen3::Eigen)
set_target_properties(eigen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# For the Eigen demo we want to explicitly use C++11, requiring Eigen aligned
# allocators.
set_target_properties(eigen PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
# TODO Cross platform flag.
#    COMPILE_FLAGS "-march=native"
)
