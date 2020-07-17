add_executable(nano_pico nano_pico.cpp)
target_link_libraries(nano_pico PUBLIC pico_common nanoflann::nanoflann)
set_target_properties(nano_pico
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
