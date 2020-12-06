# From CMake version 3.11, many of the find_package() directory structures for
# XxxConfig.cmake or FindXxx.cmake are the same for Win, Lin, etc.
cmake_minimum_required(VERSION 3.11)

project(pico_tree
    LANGUAGES CXX
    VERSION 0.5.1
    DESCRIPTION "PicoTree is a small C++ header only library for range searches and nearest neighbor searches using a KdTree."
    HOMEPAGE_URL "https://github.com/Jaybro/pico_tree")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(PROJECT_PACKAGE_NAME "PicoTree")
set(PROJECT_SOURCE_RELATIVE "src")
set(PROJECT_SOURCE_ABSOLUTE
    "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_SOURCE_RELATIVE}")

include(GNUInstallDirs)
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
# The subdirectory contains target_sources. These always seem to be made
# absolute and cannot be used for exporting the interface library.
# Error: "INTERFACE_SOURCES property contains path"
# Kept for now should a solution be found later.
# add_subdirectory(${PROJECT_SOURCE_RELATIVE})

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
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
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

install(DIRECTORY "${PROJECT_SOURCE_ABSOLUTE}/${PROJECT_NAME}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

################################################################################
# Examples, unit tests and documentation below.
################################################################################

option(BUILD_EXAMPLES "Enable the creation of PicoTree examples." ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

find_package(GTest QUIET)
if(GTEST_FOUND)
    include(CTest)
    if(BUILD_TESTING)
        # Tests are dependent on some common code.
        if(NOT TARGET pico_common)
            add_subdirectory(examples/pico_common)
        endif()

        # Unit tests
        set(TEST_TARGET_NAME ${PROJECT_NAME}_test)
        enable_testing()
        include(GoogleTest)
        add_executable(${TEST_TARGET_NAME})
        set_target_properties(${TEST_TARGET_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        add_subdirectory(test)
        include_directories(${CMAKE_CURRENT_LIST_DIR}/test)
        gtest_add_tests(
            TARGET ${TEST_TARGET_NAME}
            TEST_LIST ${TEST_TARGET_NAME}_list
        )

        target_link_libraries(${TEST_TARGET_NAME} ${PROJECT_NAME} pico_common GTest::GTest GTest::Main)
    endif()
    message(STATUS "GTest found. Unit test build: ${BUILD_TESTING}.")
else()
    message(STATUS "GTest not found. Unit tests will not be build.")
endif()

find_package(Doxygen QUIET)
include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(BUILD_DOCUMENTATION "Build documentation with Doxygen." ON "DOXYGEN_FOUND" OFF)
if(BUILD_DOCUMENTATION)
    set(DOC_TARGET_NAME ${PROJECT_NAME}_doc)
    # Hide the internal namespace from the documentation.
    # set(DOXYGEN_EXCLUDE_SYMBOLS "internal")

    doxygen_add_docs(
        ${DOC_TARGET_NAME}
        ${PROJECT_SOURCE_ABSOLUTE})

    message(STATUS "Doxygen found. Documentation can be build as: make ${DOC_TARGET_NAME}")
else()
    message(STATUS "Doxygen not found. Documentation cannot be build.")
endif()
