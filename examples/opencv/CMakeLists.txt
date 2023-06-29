add_executable(opencv opencv.cpp)
set_default_target_properties(opencv)
target_link_libraries(opencv PUBLIC pico_toolshed ${OpenCV_LIBS})
