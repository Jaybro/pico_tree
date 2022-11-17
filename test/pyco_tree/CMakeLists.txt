if(BUILD_BINDINGS)
    file(GLOB PY_SOURCES_ABS "${CMAKE_CURRENT_LIST_DIR}/" *.py)

    # TODO These don't get deleted when running: make clean
    add_custom_target(python_tests ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${PY_SOURCES_ABS} ${CMAKE_BINARY_DIR}/py)
endif()
