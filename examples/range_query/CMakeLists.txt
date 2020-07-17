add_executable(range_query range_query.cpp)
target_link_libraries(range_query PUBLIC pico_common)
set_target_properties(range_query
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
