add_library(pico_common INTERFACE)
target_include_directories(pico_common INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(pico_common INTERFACE cxx_std_11)
target_link_libraries(pico_common INTERFACE PicoTree::PicoTree)
target_sources(pico_common
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/pico_adaptor.hpp
        ${CMAKE_CURRENT_LIST_DIR}/point.hpp
        ${CMAKE_CURRENT_LIST_DIR}/scoped_timer.hpp
)
