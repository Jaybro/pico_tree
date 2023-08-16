add_library(pico_understory INTERFACE)
target_include_directories(pico_understory INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(pico_understory INTERFACE PicoTree::PicoTree)
target_sources(pico_understory
    INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/cover_tree_base.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/cover_tree_builder.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/cover_tree_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/cover_tree_node.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/cover_tree_search.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/rkd_tree_builder.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/rkd_tree_hh_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/rkd_tree_search.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/internal/static_buffer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/cover_tree.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/metric.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pico_understory/kd_forest.hpp
)
