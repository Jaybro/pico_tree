add_library(pico_toolshed INTERFACE)
target_include_directories(pico_toolshed INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(pico_toolshed INTERFACE PicoTree::PicoTree)
target_sources(pico_toolshed
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/pico_toolshed/dynamic_space.hpp
        ${CMAKE_CURRENT_LIST_DIR}/pico_toolshed/point.hpp
        ${CMAKE_CURRENT_LIST_DIR}/pico_toolshed/scoped_timer.hpp
)
