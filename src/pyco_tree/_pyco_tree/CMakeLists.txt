# The C++ module name is chosen to be _pyco_tree.
# PEP-8
# When an extension module written in C or C++ has an accompanying Python
# module that provides a higher level (e.g. more object oriented) interface,
# the C/C++ module has a leading underscore (e.g. _socket).

# Configuring using MinGW will fail when calling pybind11_add_module if
# $ENV{MSYSTEM} is not defined. This may happen while directly building from an
# IDE such as Visual Studio Code. pybind11_add_module seems to assume an MSYS
# environment when building in Windows?
pybind11_add_module(_pyco_tree)
target_sources(_pyco_tree
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/_pyco_tree.cpp
        ${CMAKE_CURRENT_LIST_DIR}/def_core.cpp
        ${CMAKE_CURRENT_LIST_DIR}/def_kd_tree.cpp
)
target_link_libraries(_pyco_tree PRIVATE PicoTree::PicoTree OpenMP::OpenMP_CXX)
set_target_properties(_pyco_tree
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
