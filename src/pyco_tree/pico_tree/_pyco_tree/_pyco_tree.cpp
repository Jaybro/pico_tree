#include <pybind11/pybind11.h>

#include <iostream>

#include "darray.hpp"
#include "def_core.hpp"
#include "def_darray.hpp"
#include "def_kd_tree.hpp"

PYBIND11_MODULE(_pyco_tree, m) {
  m.doc() =
      R"ptdoc(
PicoTree: a module for fast nearest neighbor and range searches using a
KdTree. It wraps the C++ PicoTree library.
)ptdoc";

  // Registered dtypes.
  PYBIND11_NUMPY_DTYPE(pyco_tree::Neighborf, index, distance);
  PYBIND11_NUMPY_DTYPE(pyco_tree::Neighbord, index, distance);

  pyco_tree::DefDArray(m);
  pyco_tree::DefKdTree(m);
}
