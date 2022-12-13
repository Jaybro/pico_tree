#!/usr/bin/env python3

import sys
from skbuild import setup


def compile_cxx_flags():
    cxx_flags = []

    if sys.platform == 'win32':
        # Old versions of CPython have a bug where "hypot" is defined as
        # "_hypot". This definition conflicts with the one from math.h and as a
        # result causes the following compile error in cmath using MinGW:
        #   cmath:1121:11: error: '::hypot' has not been declared
        #       using ::hypot;
        # Issue and backport reference: https://github.com/python/cpython/pull/11283
        # Issue solved: https://github.com/python/cpython/blob/v3.7.3/PC/pyconfig.h
        python_version = sys.version_info[:3]
        if python_version < (3, 7, 3):
            cxx_flags.append('-D_hypot=hypot')

    return cxx_flags


def compile_cmake_args():
    cmake_args = []

    cxx_flags = compile_cxx_flags()
    if cxx_flags:
        cmake_args.append('-DCMAKE_CXX_FLAGS="' + ' '.join(cxx_flags) + '"')

    return cmake_args


setup(name='pico_tree',
      # The same as the CMake project version.
      version='0.7.4',
      description='PicoTree Python Bindings',
      author='Jonathan Broere',
      url='https://github.com/Jaybro/pico_tree',
      license='MIT',
      packages=['pico_tree'],
      package_dir={'': 'src/pyco_tree'},
      cmake_install_dir='src/pyco_tree/pico_tree',
      cmake_args=compile_cmake_args(),
      python_requires='>=3.7',
      install_requires=['numpy'],
      )
