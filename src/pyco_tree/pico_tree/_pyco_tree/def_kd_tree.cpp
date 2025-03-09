#include "def_kd_tree.hpp"

#include "darray.hpp"
#include "def_core.hpp"
#include "kd_tree.hpp"

namespace py = pybind11;

namespace pyco_tree {

void def_kd_tree(py::module& m) {
  using size_t = typename kd_tree::size_type;

  py::enum_<metric_t>(m, "Metric")
      .value("L1", metric_t::l1)
      .value("L2Squared", metric_t::l2_squared)
      .value("LInf", metric_t::linf);

  py::class_<kd_tree>(m, "KdTree", py::buffer_protocol())
      .def(
          py::init<py::array, metric_t, size_t>(),
          // We keep the input points alive until the kd_tree is gone.
          // Nurse: 1 = Implicit first argument: kd_tree.
          // Patient: 2 = Second function argument: NumPy array.
          py::keep_alive<1, 2>(),
          py::arg("pts"),
          py::arg("metric"),
          py::arg("max_leaf_size"),
          "Create a new KdTree from given points and maximum leaf size.")
      // The kd_tree can be used as a read-only buffer. It helps to check how
      // the data is interpreted.
      .def_buffer(
          [](kd_tree const& t) -> py::buffer_info { return t.request(); })
      .def("__repr__", [](kd_tree const& t) { return t.repr(); })
      .def_property_readonly(
          "dtype_index",
          [](kd_tree const& t) { return t.dtype_index(); },
          "Get the dtype of indices returned by this tree.")
      .def_property_readonly(
          "dtype_scalar",
          [](kd_tree const& t) { return t.dtype_scalar(); },
          "Get the dtype of scalars returned by this tree.")
      .def_property_readonly(
          "dtype_neighbor",
          [](kd_tree const& t) { return t.dtype_neighbor(); },
          "Get the dtype of neighbors returned by this tree.")
      .def_property_readonly(
          "sdim", &kd_tree::sdim, "Get the spatial dimension of the KdTree.")
      .def_property_readonly(
          "npts", &kd_tree::npts, "Get the number of points of the KdTree.")
      // Because of the ambiguity that arises from function overloading, we have
      // to show which function we mean by using a static_cast.
      .def(
          "metric",
          static_cast<py::float_ (kd_tree::*)(py::float_ const) const>(
              &kd_tree::metric),
          py::arg("scalar"),
          "Return a scalar with the metric applied.")
      .def(
          "search_knn",
          static_cast<void (kd_tree::*)(
              py::array const, py::int_ const, py::array) const>(
              &kd_tree::search_knn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("nns").noconvert().none(false),
          R"ptdoc(
Search the k nearest neighbors for each of the input points and store
the result in the specified output array. The output will be resized
when its shape is not (npts, k). If resized, its memory layout will be
the same as that of the input.
)ptdoc")
      .def(
          "search_knn",
          static_cast<py::array (kd_tree::*)(py::array const, py::int_ const)
                          const>(&kd_tree::search_knn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          R"ptdoc(
Search the k nearest neighbors for each of the input points. The memory
layout of the output will be the same as that of the input.
)ptdoc")
      .def(
          "search_knn",
          static_cast<void (kd_tree::*)(
              py::array const, py::int_ const, py::float_ const, py::array)
                          const>(&kd_tree::search_knn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("e").none(false),
          py::arg("nns").noconvert().none(false),
          R"ptdoc(
Search the k approximate nearest neighbors for each of the input points
and store the result in the specified output array. The output will be
resized when its shape is not (npts, k). If resized, its memory layout
will be the same as that of the input.
)ptdoc")
      .def(
          "search_knn",
          static_cast<py::array (kd_tree::*)(
              py::array const, py::int_ const, py::float_ const) const>(
              &kd_tree::search_knn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("e").none(false),
          R"ptdoc(
Search the k approximate nearest neighbors for each of the input
points. The memory layout of the output will be the same as that of the
input.
)ptdoc")
      .def(
          "search_radius",
          static_cast<void (kd_tree::*)(
              py::array const, py::float_ const, darray&, bool const) const>(
              &kd_tree::search_radius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("nns").noconvert().none(false),
          py::arg("sort").none(false) = false,
          R"ptdoc(
Search for all neighbors within a radius of each of the input points
and store the result in the specified output.
)ptdoc")
      .def(
          "search_radius",
          static_cast<darray (kd_tree::*)(
              py::array const, py::float_ const, bool const) const>(
              &kd_tree::search_radius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("sort").none(false) = false,
          R"ptdoc(
Search for all neighbors within a radius of each of the input points.
)ptdoc")
      .def(
          "search_radius",
          static_cast<void (kd_tree::*)(
              py::array const,
              py::float_ const,
              py::float_ const,
              darray&,
              bool const) const>(&kd_tree::search_radius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("e").none(false),
          py::arg("nns").noconvert().none(false),
          py::arg("sort").none(false) = false,
          R"ptdoc(
Search for the approximate neighbors within a radius of each of the
input points and store the result in the specified output.
)ptdoc")
      .def(
          "search_radius",
          static_cast<darray (kd_tree::*)(
              py::array const, py::float_ const, py::float_ const, bool const)
                          const>(&kd_tree::search_radius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("e").none(false),
          py::arg("sort").none(false) = false,
          R"ptdoc(
Search for the approximate neighbors within a radius of each of the
input points.
)ptdoc")
      .def(
          "search_box",
          static_cast<void (kd_tree::*)(py::array const, darray&) const>(
              &kd_tree::search_box),
          py::arg("boxes").noconvert().none(false),
          py::arg("indices").noconvert().none(false),
          R"ptdoc(
Search for all points within each of the axis aligned input boxes and
store the result in the specified output.
)ptdoc")
      .def(
          "search_box",
          static_cast<darray (kd_tree::*)(py::array const) const>(
              &kd_tree::search_box),
          py::arg("boxes").noconvert().none(false),
          "Search for all points within each of the axis aligned input boxes.");

  m.def(
      "load_kd_tree",
      &load_kd_tree,
      py::return_value_policy::move,
      py::arg("pts"),
      py::arg("filename"),
      "Load a KdTree from file given points and a filename.");

  m.def(
      "save_kd_tree",
      &save_kd_tree,
      py::arg("tree"),
      py::arg("filename"),
      "Save a KdTree to file given a tree and a filename.");
}

}  // namespace pyco_tree
