add_subdirectory(pico_toolshed)

# Sunlight may not reach every tree equally.
# Trees or other search structures that end up in the understory can still be
# interesting to others, they can be used for comparisons, or perhaps promoted
# later.
add_subdirectory(pico_understory)

add_subdirectory(kd_tree)

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found. Building Eigen example.")
    add_subdirectory(eigen)
else()
    message(STATUS "Eigen3 not found. Eigen example skipped.")
endif()

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Building OpenCV example.")
    add_subdirectory(opencv)
else()
    message(STATUS "OpenCV not found. OpenCV example skipped.")
endif()

function(HAS_CXX_COMPILE_FEATURE FEATURE HAS_FEATURE)
    foreach(i ${CMAKE_CXX_COMPILE_FEATURES})
        if(${i} STREQUAL ${FEATURE})
            set(${HAS_FEATURE} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${HAS_FEATURE} FALSE PARENT_SCOPE)
endfunction()

set(HAS_CPP17_FEATURE FALSE)
has_cxx_compile_feature(cxx_std_17 HAS_CPP17_FEATURE)
find_package(benchmark QUIET)
if(benchmark_FOUND AND HAS_CPP17_FEATURE)
    message(STATUS "benchmark and C++17 found. Building PicoTree benchmarks.")
    add_subdirectory(benchmark)
else()
    message(STATUS "benchmark or C++17 not found. PicoTree benchmarks skipped.")
endif()

# The Python examples only get copied when the bindings module will be build.
if(TARGET _pyco_tree)
    add_subdirectory(python)
endif()
