# From CMake version 3.11, many of the find_package() directory structures for
# XxxConfig.cmake or FindXxx.cmake are the same for Win, Lin, etc.
cmake_minimum_required(VERSION 3.11)

project(pico_tree LANGUAGES CXX)

set(PROJECT_SOURCE_RELATIVE "src")
set(PROJECT_SOURCE_ABSOLUTE
    "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_SOURCE_RELATIVE}"
)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_ABSOLUTE})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)
add_subdirectory(${PROJECT_SOURCE_RELATIVE})

option(BUILD_EXAMPLES "Enable the creation of pico_tree examples." ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CMakeDependentOption)
find_package(GTest)
CMAKE_DEPENDENT_OPTION(BUILD_TESTING "Build unit tests with GTest." ON "GTEST_FOUND" OFF)
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
    add_subdirectory(test)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/test)
    gtest_add_tests(
        TARGET ${TEST_TARGET_NAME}
        TEST_LIST ${TEST_TARGET_NAME}_list
    )

    target_link_libraries(${TEST_TARGET_NAME} ${PROJECT_NAME} pico_common GTest::GTest GTest::Main)
endif()

find_package(Doxygen)
CMAKE_DEPENDENT_OPTION(BUILD_DOCUMENTATION "Build documentation with Doxygen." ON "DOXYGEN_FOUND" OFF)
if(BUILD_DOCUMENTATION)
    set(DOC_TARGET_NAME ${PROJECT_NAME}_doc)
    # Hide the internal namespace from the documentation.
    # set(DOXYGEN_EXCLUDE_SYMBOLS "internal")

    doxygen_add_docs(
        ${DOC_TARGET_NAME} 
        ${PROJECT_SOURCE_ABSOLUTE})
endif()
