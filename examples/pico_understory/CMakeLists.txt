add_library(pico_understory INTERFACE)
target_include_directories(pico_understory INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(pico_understory INTERFACE cxx_std_11)
target_link_libraries(pico_understory INTERFACE PicoTree::PicoTree)
target_sources(pico_understory
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/pico_understory/cover_tree.hpp
)
