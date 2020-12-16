include(GoogleTest)

set(TEST_TARGET_NAME ${PROJECT_NAME}_test)
add_executable(${TEST_TARGET_NAME})
set_target_properties(${TEST_TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
include_directories(${CMAKE_CURRENT_LIST_DIR})
target_sources(${TEST_TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/kd_tree_test.cpp
)
target_link_libraries(${TEST_TARGET_NAME} ${PROJECT_NAME} pico_common GTest::GTest GTest::Main)

gtest_add_tests(
    TARGET ${TEST_TARGET_NAME}
    TEST_LIST ${TEST_TARGET_NAME}_list
)
