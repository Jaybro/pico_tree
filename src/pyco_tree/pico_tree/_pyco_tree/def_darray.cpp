#include "def_darray.hpp"

#include "darray.hpp"

namespace py = pybind11;

namespace pyco_tree {

void DefDArray(pybind11::module* m) {
  py::class_<DArray>(*m, "DArray")
      .def(
          py::init<py::dtype>(),
          py::arg("dtype").none(false),
          "Create a DArray from a numpy dtype.")
      .def(
          "__iter__",
          [](DArray& a) { return py::make_iterator(a.begin(), a.end()); },
          py::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](DArray& a, DArray::difference_type i) {
            if (i < 0) {
              i += a.size();
            }

            if (i < 0 || static_cast<DArray::size_type>(i) >= a.size()) {
              throw py::index_error();
            }

            return a[static_cast<DArray::size_type>(i)];
          },
          py::arg("i").none(false),
          py::keep_alive<0, 1>())
      .def(
          "__bool__",
          [](DArray const& a) -> bool { return !a.empty(); },
          "Check whether the list is nonempty")
      .def("__len__", &DArray::size)
      .def_property_readonly("dtype", &DArray::dtype);
}

}  // namespace pyco_tree
