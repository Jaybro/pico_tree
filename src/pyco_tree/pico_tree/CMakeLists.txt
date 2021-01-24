# The C++ module name equals _pyco_tree in accordance with PEP-8:
# When an extension module written in C or C++ has an accompanying Python
# module that provides a higher level (e.g. more object oriented) interface,
# the C/C++ module has a leading underscore (e.g. _socket).

# pybind11_add_module seems to assume an MSYS environment when building on a
# Windows machine?
# Configuring using MinGW Makefiles will fail when calling pybind11_add_module
# if $ENV{MSYSTEM} is not defined. This may happen while directly building from
# an IDE such as Visual Studio Code. Solve by adding the following to the
# settings.json of VSC:
#
# "cmake.environment": {
#     "MSYSTEM": "MINGW64"
# },
pybind11_add_module(_pyco_tree MODULE)
target_sources(_pyco_tree
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/_pyco_tree/_pyco_tree.cpp
        ${CMAKE_CURRENT_LIST_DIR}/_pyco_tree/def_darray.cpp
        ${CMAKE_CURRENT_LIST_DIR}/_pyco_tree/def_kd_tree.cpp
)
target_link_libraries(_pyco_tree PRIVATE PicoTree::PicoTree OpenMP::OpenMP_CXX)

set_target_properties(_pyco_tree
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyco_tree"
)

file(GLOB PY_SOURCES_ABS "${CMAKE_CURRENT_LIST_DIR}" *.py)
# TODO These don't get deleted when running: make clean
add_custom_command(TARGET _pyco_tree POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_SOURCES_ABS} ${CMAKE_BINARY_DIR}/pyco_tree)

if(SKBUILD)
    install(TARGETS _pyco_tree LIBRARY DESTINATION .)
endif()
