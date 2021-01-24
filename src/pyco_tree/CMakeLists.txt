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
    # scikit-build incorrectly detects Python under MSYS2 / MinGW.
    # https://github.com/scikit-build/scikit-build/blob/master/skbuild/cmaker.py#L358
    # The function get_python_library inside cmaker.py uses:
    #   python_library = sysconfig.get_config_var('LIBRARY')
    # Which returns libpython3.8.a even though the static library is not
    # provided via MSYS2.
    # https://github.com/msys2/MINGW-packages/issues/3562
    # scikit-build otherwise doesn't seem to work well with
    # find_package(Python). The PYTHON_LIBRARY flag conflicts with the 
    # Python_LIBRARY one from find_package().
    # The scikit-build provided flags are unset to solve both previously
    # mentioned issues.
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHON_VERSION_STRING CACHE)
    unset(PYTHON_INCLUDE_DIR CACHE)
    unset(PYTHON_LIBRARY CACHE)
    
    # pybind11 also looks for Python.
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    find_package(OpenMP)

    add_subdirectory(pico_tree)
    message(STATUS "Building PycoTree Python Module via scikit-build.")
endif()
