include(GNUInstallDirs)

################################################################################
# PicoTree library.
################################################################################

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_PACKAGE_NAME}::${PROJECT_PACKAGE_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
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

################################################################################
# Generation and installation of Targets.cmake, Config.cmake,
# ConfigVersion.cmake and PicoTree itself.
################################################################################

set(PROJECT_PACKAGE_TARGETS_NAME ${PROJECT_PACKAGE_NAME}Targets)
set(PROJECT_PACKAGE_CONFIG_VERSION_NAME ${PROJECT_PACKAGE_NAME}ConfigVersion)
set(PROJECT_PACKAGE_CONFIG_NAME ${PROJECT_PACKAGE_NAME}Config)

# Not adding INCLUDE DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# See target_include_directories
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_PACKAGE_TARGETS_NAME})

export(EXPORT ${PROJECT_PACKAGE_TARGETS_NAME}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_PACKAGE_TARGETS_NAME}.cmake"
    NAMESPACE ${PROJECT_PACKAGE_NAME}::)

set(PACKAGE_INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_PACKAGE_NAME}/cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_PACKAGE_CONFIG_VERSION_NAME}.cmake"
    VERSION ${PROJECT_VERSION}
    # Beta library will use the minor version for compatibility.
    COMPATIBILITY SameMinorVersion)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_PACKAGE_CONFIG_NAME}.cmake"
    INSTALL_DESTINATION ${PACKAGE_INSTALL_DESTINATION})

install(EXPORT ${PROJECT_PACKAGE_TARGETS_NAME}
    FILE "${PROJECT_PACKAGE_TARGETS_NAME}.cmake"
    NAMESPACE ${PROJECT_PACKAGE_NAME}::
    DESTINATION ${PACKAGE_INSTALL_DESTINATION})

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_PACKAGE_CONFIG_VERSION_NAME}.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_PACKAGE_CONFIG_NAME}.cmake"
    DESTINATION ${PACKAGE_INSTALL_DESTINATION})

install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
