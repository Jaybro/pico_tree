function(set_default_target_properties TARGET_NAME)
    set_target_properties(${TARGET_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

function(has_cxx_compile_feature VAR_HAS_FEATURE FEATURE)
    foreach(i ${CMAKE_CXX_COMPILE_FEATURES})
        if(${i} STREQUAL ${FEATURE})
            set(${VAR_HAS_FEATURE} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${VAR_HAS_FEATURE} FALSE PARENT_SCOPE)
endfunction()
