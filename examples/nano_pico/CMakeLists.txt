add_executable(nano_pico nano_pico.cpp)
target_link_libraries(nano_pico PUBLIC pico_common nanoflann::nanoflann)
