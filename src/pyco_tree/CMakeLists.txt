if(NOT SKBUILD)
    find_package(Python COMPONENTS Interpreter Development QUIET)
    find_package(pybind11 CONFIG QUIET)
    find_package(OpenMP QUIET)

    if (Python_FOUND AND pybind11_FOUND AND OpenMP_FOUND)
        add_subdirectory(pico_tree)
        message(STATUS "Python, pybind11 and OpenMP found. Building PycoTree Python Module.")
    else()
        message(STATUS "Python, pybind11 or OpenMP not found. PycoTree Python Module skipped.")
    endif()
else()
    # scikit-build incorrectly detects Python under MSYS2 / MinGW which it then
    # "manually" specifies to CMake. To solve this issue we want to ignore some
    # of the manually set flags and use find_package(Python) to correct the
    # issue.
    # The problem starts here:
    # https://github.com/scikit-build/scikit-build/blob/master/skbuild/cmaker.py#L358
    # The function get_python_library inside cmaker.py uses:
    #   python_library = sysconfig.get_config_var('LIBRARY')
    # It returns a static library, e.g. libpython3.8.a, even though it isn't
    # provided via MSYS2.
    # https://github.com/msys2/MINGW-packages/issues/3562
    
    # Ignoring manually specified flags results in a warning. Here follows
    # fake usage of the flags.
    set(IGNORE_WARNING "${PYTHON_EXECUTABLE}${PYTHON_INCLUDE_DIR}${PYTHON_LIBRARY}")
    
    # Unset the flags to solve the previously mentioned issue.
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHON_INCLUDE_DIR CACHE)
    unset(PYTHON_LIBRARY CACHE)
    
    # We want to find the Python version specified by scikit build. This makes
    # the build process transparent, but also prevents issues in environments
    # where multiple versions of Python are installed.
    find_package(Python ${PYTHON_VERSION_STRING} EXACT COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    find_package(OpenMP)

    add_subdirectory(pico_tree)
    message(STATUS "Building PycoTree Python Module via scikit-build.")
endif()
