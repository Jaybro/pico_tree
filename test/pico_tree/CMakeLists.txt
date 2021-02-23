include(GoogleTest)

find_package(Eigen3 QUIET)

set(TEST_TARGET_NAME ${PROJECT_NAME}_test)
add_executable(${TEST_TARGET_NAME})
set_target_properties(${TEST_TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# The library should be compliant with the C++11 standard.
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(TEST_TARGET_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cover_tree_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kd_tree_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/metric_test.cpp
)

# gtest_add_tests fails on generator expressions like:
#$<$<TARGET_EXISTS:Eigen3::Eigen>:${CMAKE_CURRENT_LIST_DIR}/eigen.cpp>
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found. Building Eigen unit tests.")
    set(TEST_TARGET_SOURCES
        ${TEST_TARGET_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/eigen_test.cpp
    )
else()
    message(STATUS "Eigen3 not found. Eigen unit tests skipped.")
endif()

target_sources(${TEST_TARGET_NAME} PRIVATE ${TEST_TARGET_SOURCES})
target_link_libraries(${TEST_TARGET_NAME}
    ${PROJECT_NAME}
    pico_common
    GTest::GTest
    GTest::Main
    $<$<TARGET_EXISTS:Eigen3::Eigen>:Eigen3::Eigen>
)

gtest_add_tests(
    TARGET ${TEST_TARGET_NAME}
    TEST_LIST ${TEST_TARGET_NAME}_list
)
