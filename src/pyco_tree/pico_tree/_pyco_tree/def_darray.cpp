#include "def_darray.hpp"

#include "darray.hpp"

namespace py = pybind11;

namespace pyco_tree {

void def_darray(pybind11::module& m) {
  py::class_<darray>(
      m,
      "DArray",
      R"ptdoc(
A class whose instance represents a dynamic array of numpy arrays.
Resizing an array and its contents is not possible but the values of
the numpy arrays may be modified. The numpy arrays always have a single
dimension and they don't have to be of equal size.
)ptdoc")
      .def(
          py::init([](py::object dtype) {
            return darray(py::dtype::from_args(dtype));
          }),
          py::arg("dtype").none(false),
          R"ptdoc(
Create a darray from any object that can be used to construct a numpy
dtype.
)ptdoc")
      .def(
          "__iter__",
          [](darray& a) { return py::make_iterator(a.begin(), a.end()); },
          py::keep_alive<0, 1>(),
          "Return an iterator over the contained ndarrays.")
      .def(
          "__getitem__",
          [](darray& a, darray::difference_type i) {
            if (i < 0) {
              i += a.size();
            }

            if (i < 0 || static_cast<darray::size_type>(i) >= a.size()) {
              throw py::index_error();
            }

            return a[static_cast<darray::size_type>(i)];
          },
          py::arg("i").none(false),
          py::keep_alive<0, 1>(),
          "x.__getitem__(y) <==> x[y].")
      // Slicing protocol
      .def(
          "__getitem__",
          [](darray const& a, py::slice slice) -> darray {
            std::size_t start, stop, step, slice_length;

            if (!slice.compute(a.size(), &start, &stop, &step, &slice_length))
              throw py::error_already_set();

            return a.copy(start, step, slice_length);
          },
          py::arg("s"),
          "x.__getitem__(y) <==> x[y].")
      .def("__len__", &darray::size, "Return len(self).")
      .def_property_readonly(
          "dtype",
          &darray::dtype,
          "Get the dtype of neighbors returned by this array.");
}

}  // namespace pyco_tree
