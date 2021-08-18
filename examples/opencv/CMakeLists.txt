add_executable(opencv opencv.cpp)
set_default_target_properties(opencv)
target_link_libraries(opencv PUBLIC pico_toolshed ${OpenCV_LIBS})
set_target_properties(opencv
    PROPERTIES
# For the OpenCV demo we want to explicitly use C++11.
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
