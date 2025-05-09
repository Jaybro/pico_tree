include(GoogleTest)

set(TEST_TARGET_NAME ${PROJECT_NAME}_test)
add_executable(${TEST_TARGET_NAME})
set_default_target_properties(${TEST_TARGET_NAME})
set_target_properties(${TEST_TARGET_NAME}
    PROPERTIES

    # The library should be compliant with the version of the standard that is
    # minimally required by this library.
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(TEST_TARGET_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/box_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cover_tree_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/distance_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kd_tree_builder_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kd_tree_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/metric_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/point_map_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/space_map_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/space_map_traits_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vector_traits_test.cpp
)

# gtest_add_tests fails on generator expressions like:
# $<$<TARGET_EXISTS:Eigen3::Eigen>:${CMAKE_CURRENT_LIST_DIR}/eigen.cpp>
find_package(Eigen3 QUIET)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found. Building Eigen unit tests.")
    set(TEST_TARGET_SOURCES
        ${TEST_TARGET_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/eigen3_traits_test.cpp
    )
else()
    message(STATUS "Eigen3 not found. Eigen unit tests skipped.")
endif()

find_package(OpenCV QUIET)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Building OpenCV unit tests.")
    set(TEST_TARGET_SOURCES
        ${TEST_TARGET_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/opencv_traits_test.cpp
    )
else()
    message(STATUS "OpenCV not found. OpenCV unit tests skipped.")
endif()

target_sources(${TEST_TARGET_NAME} PRIVATE ${TEST_TARGET_SOURCES})
target_link_libraries(${TEST_TARGET_NAME}
    ${PROJECT_NAME}
    pico_toolshed

    # Testing the understory structures here as well.
    pico_understory
    GTest::GTest
    GTest::Main
    $<$<TARGET_EXISTS:Eigen3::Eigen>:Eigen3::Eigen>
    $<$<TARGET_EXISTS:opencv_core>:opencv_core>
)

gtest_add_tests(
    TARGET ${TEST_TARGET_NAME}
    TEST_LIST ${TEST_TARGET_NAME}_list
)
