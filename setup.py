#!/usr/bin/env python3

from skbuild import setup

setup(name='pico_tree',
      # The version of PycoTree follows that of PicoTree.
      version='0.6.0',
      description='PicoTree Python Bindings',
      author='Jonathan Broere',
      # url='https://github.com/Jaybro/pico_tree',
      license="MIT",
      packages=["pico_tree"],
      package_dir={"": "src/pyco_tree"},
      cmake_install_dir="src/pyco_tree/pico_tree",
      )
