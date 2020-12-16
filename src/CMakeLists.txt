add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_PACKAGE_NAME}::${PROJECT_PACKAGE_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_ABSOLUTE}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
# Language standard above 11 should also be fine.
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME ${PROJECT_PACKAGE_NAME})
target_compile_options(${PROJECT_NAME} INTERFACE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4>)

# The target_sources always seem to be made absolute and cannot be used for
# exporting the interface library.
# Error: "INTERFACE_SOURCES property contains path"
# Kept for now should a solution be found later.
# target_sources(${PROJECT_NAME}
#     INTERFACE
#         ${CMAKE_CURRENT_LIST_DIR}/pico_tree/core.hpp
#         ${CMAKE_CURRENT_LIST_DIR}/pico_tree/eigen.hpp
#         ${CMAKE_CURRENT_LIST_DIR}/pico_tree/kd_tree.hpp
# )
