# From CMake version 3.11, many of the find_package() directory structures for
# XxxConfig.cmake or FindXxx.cmake are the same for Win, Lin, etc.
cmake_minimum_required(VERSION 3.11)

project(nanotree LANGUAGES CXX)

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)

set(PROJECT_LIBS 
    glog::glog
    Eigen3::Eigen
    nanoflann::nanoflann
)

set(PROJECT_SOURCE_RELATIVE "src")
set(PROJECT_SOURCE_ABSOLUTE
    "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_SOURCE_RELATIVE}"
)
# Most things come from the linked targets. Not the following includes.
set(PROJECT_INCLUDES 
    ${PROJECT_SOURCE_ABSOLUTE}
)

add_library(${PROJECT_NAME} INTERFACE)
add_subdirectory(${PROJECT_SOURCE_RELATIVE})
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_INCLUDES})
target_link_libraries(${PROJECT_NAME} INTERFACE ${PROJECT_LIBS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

include(CTest)
if(BUILD_TESTING)
    set(TEST_TARGET_NAME ${PROJECT_NAME}_test)
    enable_testing()
    include(GoogleTest)
    find_package(GTest REQUIRED)
    #add_executable(${TEST_TARGET_NAME})
    #add_subdirectory(test)
    #include_directories(${CMAKE_CURRENT_LIST_DIR}/test)
    #gtest_add_tests(
    #    TARGET ${TEST_TARGET_NAME}
    #    TEST_LIST ${TEST_TARGET_NAME}_list
    #)
    #target_link_libraries(${TEST_TARGET_NAME} ${PROJECT_NAME} GTest::GTest GTest::Main)
endif()

include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(BUILD_DOCUMENTATION "Build documentation with Doxygen." ON "DOXYGEN_FOUND" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    set(DOC_TARGET_NAME ${PROJECT_NAME}_doc)
    # Hide the internal namespace from the documentation.
    set(DOXYGEN_EXCLUDE_SYMBOLS "internal")

    doxygen_add_docs(
        ${DOC_TARGET_NAME} 
        ${PROJECT_SOURCE_ABSOLUTE})
endif()
