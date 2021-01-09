# From CMake version 3.11, many of the find_package() directory structures for
# XxxConfig.cmake or FindXxx.cmake are the same for Win, Lin, etc.
cmake_minimum_required(VERSION 3.11)

project(pico_tree
    LANGUAGES CXX
    VERSION 0.5.2
    DESCRIPTION "PicoTree is a small C++ header only library for range searches and nearest neighbor searches using a KdTree."
    HOMEPAGE_URL "https://github.com/Jaybro/pico_tree")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

################################################################################
# PicoTree, examples, unit tests and documentation.
################################################################################

set(PROJECT_PACKAGE_NAME "PicoTree")
set(PROJECT_SOURCE_RELATIVE "src")
set(PROJECT_SOURCE_ABSOLUTE
    "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_SOURCE_RELATIVE}")

add_subdirectory(${PROJECT_SOURCE_RELATIVE})

option(BUILD_EXAMPLES "Enable the creation of PicoTree examples." ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

find_package(GTest QUIET)
if(GTEST_FOUND)
    include(CTest)
    message(STATUS "GTest found. Unit test build: ${BUILD_TESTING}.")
    if(BUILD_TESTING)
        # Tests are dependent on some common code.
        if(NOT TARGET pico_common)
            add_subdirectory(examples/pico_common)
        endif()

        enable_testing()
        add_subdirectory(test)
    endif()
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
