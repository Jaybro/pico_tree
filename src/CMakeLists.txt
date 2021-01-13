add_subdirectory(pico_tree)

find_package(Python COMPONENTS Interpreter Development QUIET)
find_package(pybind11 CONFIG QUIET)
find_package(OpenMP QUIET)

if (Python_FOUND AND pybind11_FOUND AND OpenMP_FOUND)
    add_subdirectory(pyco_tree/_pyco_tree)
    message(STATUS "Python, pybind11 and OpenMP found. Building PycoTree Python Module.")
else()
    message(STATUS "Python, pybind11 or OpenMP not found. PycoTree Python Module skipped.")
endif()
