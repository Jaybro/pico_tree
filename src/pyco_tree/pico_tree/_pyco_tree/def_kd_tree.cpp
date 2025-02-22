#include "def_kd_tree.hpp"

#include "darray.hpp"
#include "def_core.hpp"
#include "kd_tree.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename Point_>
using kd_tree_l1 = kd_tree<space_x<Point_>, pico_tree::metric_l1>;
template <typename Point_>
using kd_tree_l2_squared =
    kd_tree<space_x<Point_>, pico_tree::metric_l2_squared>;
template <typename Point_>
using kd_tree_linf = kd_tree<space_x<Point_>, pico_tree::metric_linf>;

// Fixed size 2d kd_tree
using KdTree2fL1 = kd_tree_l1<space_2f>;
using KdTree2fL2Squared = kd_tree_l2_squared<space_2f>;
using KdTree2fLInf = kd_tree_linf<space_2f>;
using KdTree2dL1 = kd_tree_l1<space_2d>;
using KdTree2dL2Squared = kd_tree_l2_squared<space_2d>;
using KdTree2dLInf = kd_tree_linf<space_2d>;

// Fixed size 3d kd_tree
using KdTree3fL1 = kd_tree_l1<space_3f>;
using KdTree3fL2Squared = kd_tree_l2_squared<space_3f>;
using KdTree3fLInf = kd_tree_linf<space_3f>;
using KdTree3dL1 = kd_tree_l1<space_3d>;
using KdTree3dL2Squared = kd_tree_l2_squared<space_3d>;
using KdTree3dLInf = kd_tree_linf<space_3d>;

// Dynamic size kd_tree
using KdTreeXfL1 = kd_tree_l1<space_xf>;
using KdTreeXfL2Squared = kd_tree_l2_squared<space_xf>;
using KdTreeXfLInf = kd_tree_linf<space_xf>;
using KdTreeXdL1 = kd_tree_l1<space_xd>;
using KdTreeXdL2Squared = kd_tree_l2_squared<space_xd>;
using KdTreeXdLInf = kd_tree_linf<space_xd>;

template <typename KdTree_>
void def_kd_tree(std::string const& name, py::module& m) {
  using kd_tree = KdTree_;
  using index = typename kd_tree::index_type;
  using scalar = typename kd_tree::scalar_type;
  using size_t = typename kd_tree::size_type;
  using metric = typename kd_tree::metric_type;
  using neighbor = typename kd_tree::neighbor_type;
  using neighborhoods = darray;

  py::class_<kd_tree>(m, name.c_str(), py::buffer_protocol())
      .def(
          py::init<py::array_t<scalar, 0>, size_t>(),
          // We keep the input points alive until the kd_tree is gone.
          // Nurse: 1 = Implicit first argument: kd_tree.
          // Patient: 2 = Second function argument: NumPy array.
          py::keep_alive<1, 2>(),
          py::arg("pts"),
          py::arg("max_leaf_size"),
          "Create a new KdTree from given points and maximum leaf size.")
      // The kd_tree can be used as a read-only buffer. It helps to check how
      // the data is interpreted.
      .def_buffer([](kd_tree& t) -> py::buffer_info {
        py::ssize_t const sdim = static_cast<py::ssize_t>(t.sdim());
        py::ssize_t const npts = static_cast<py::ssize_t>(t.npts());
        py::ssize_t const inner_stride = sizeof(scalar);
        py::ssize_t const outer_stride = inner_stride * sdim;

        // There doesn't appear to be a true read only-interface. Hence the rare
        // const_cast.
        return py::buffer_info(
            const_cast<void*>(static_cast<void const*>(t.data())),
            // Item size:
            inner_stride,
            py::format_descriptor<scalar>::format(),
            // Array/Tensor dimensions:
            2,
            // Shape:
            t.row_major() ? std::vector<py::ssize_t>{npts, sdim}
                          : std::vector<py::ssize_t>{sdim, npts},
            // Strides:
            t.row_major()
                ? std::vector<py::ssize_t>{outer_stride, inner_stride}
                : std::vector<py::ssize_t>{inner_stride, outer_stride},
            // Read only (at least from the kd_tree side of things)
            true);
      })
      .def(
          "__repr__",
          [](kd_tree const& t) {
            return "KdTree(metric=" + string_traits<metric>::type_string() +
                   ", dtype=" + string_traits<scalar>::type_string() +
                   ", sdim=" + std::to_string(t.sdim()) +
                   ", npts=" + std::to_string(t.npts()) + ")";
          })
      .def_property_readonly(
          "dtype_index",
          [](kd_tree const&) { return py::dtype::of<index>(); },
          "Get the dtype of indices returned by this tree.")
      .def_property_readonly(
          "dtype_scalar",
          [](kd_tree const&) { return py::dtype::of<scalar>(); },
          "Get the dtype of scalars returned by this tree.")
      .def_property_readonly(
          "dtype_neighbor",
          [](kd_tree const&) { return py::dtype::of<neighbor>(); },
          "Get the dtype of neighbors returned by this tree.")
      .def_property_readonly(
          "sdim", &kd_tree::sdim, "Get the spatial dimension of the KdTree.")
      .def_property_readonly(
          "npts", &kd_tree::npts, "Get the number of points of the KdTree.")
      // Because of the ambiguity that arises from function overloading, we have
      // to show which function we mean by using a static_cast.
      .def(
          "metric",
          static_cast<scalar (kd_tree::*)(scalar) const>(&kd_tree::metric),
          py::arg("scalar"),
          "Return a scalar with the metric applied.")
      .def(
          "search_knn",
          static_cast<void (kd_tree::*)(
              py::array_t<scalar, 0> const,
              size_t const,
              py::array_t<neighbor, 0>) const>(&kd_tree::search_knn),
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
          static_cast<py::array_t<neighbor, 0> (kd_tree::*)(
              py::array_t<scalar, 0> const, size_t const) const>(
              &kd_tree::search_knn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          R"ptdoc(
Search the k nearest neighbors for each of the input points. The memory
layout of the output will be the same as that of the input.
)ptdoc")
      .def(
          "search_knn",
          static_cast<void (kd_tree::*)(
              py::array_t<scalar, 0> const,
              size_t const,
              scalar const,
              py::array_t<neighbor, 0>) const>(&kd_tree::search_knn),
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
          static_cast<py::array_t<neighbor, 0> (kd_tree::*)(
              py::array_t<scalar, 0> const, size_t const, scalar const) const>(
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
              py::array_t<scalar, 0> const,
              scalar const,
              neighborhoods&,
              bool const) const>(&kd_tree::search_radius),
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
          static_cast<neighborhoods (kd_tree::*)(
              py::array_t<scalar, 0> const, scalar const, bool const) const>(
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
              py::array_t<scalar, 0> const,
              scalar const,
              scalar const,
              neighborhoods&,
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
          static_cast<neighborhoods (kd_tree::*)(
              py::array_t<scalar, 0> const,
              scalar const,
              scalar const,
              bool const) const>(&kd_tree::search_radius),
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
          static_cast<void (kd_tree::*)(
              py::array_t<scalar, 0> const, neighborhoods&) const>(
              &kd_tree::search_box),
          py::arg("boxes").noconvert().none(false),
          py::arg("indices").noconvert().none(false),
          R"ptdoc(
Search for all points within each of the axis aligned input boxes and
store the result in the specified output.
)ptdoc")
      .def(
          "search_box",
          static_cast<neighborhoods (kd_tree::*)(py::array_t<scalar, 0> const)
                          const>(&kd_tree::search_box),
          py::arg("boxes").noconvert().none(false),
          "Search for all points within each of the axis aligned input boxes.");
}

void def_kd_tree(py::module& m) {
  // TODO: This should become faster to compile.
  // TODO: Perhaps apply type erasure to reduce repetition.
  // Fixed size 2d kd_tree
  def_kd_tree<KdTree2fL1>("KdTree2fL1", m);
  def_kd_tree<KdTree2fL2Squared>("KdTree2fL2Squared", m);
  def_kd_tree<KdTree2fLInf>("KdTree2fLInf", m);
  def_kd_tree<KdTree2dL1>("KdTree2dL1", m);
  def_kd_tree<KdTree2dL2Squared>("KdTree2dL2Squared", m);
  def_kd_tree<KdTree2dLInf>("KdTree2dLInf", m);

  // Fixed size 3d kd_tree
  def_kd_tree<KdTree3fL1>("KdTree3fL1", m);
  def_kd_tree<KdTree3fL2Squared>("KdTree3fL2Squared", m);
  def_kd_tree<KdTree3fLInf>("KdTree3fLInf", m);
  def_kd_tree<KdTree3dL1>("KdTree3dL1", m);
  def_kd_tree<KdTree3dL2Squared>("KdTree3dL2Squared", m);
  def_kd_tree<KdTree3dLInf>("KdTree3dLInf", m);

  // Dynamic size kd_tree
  def_kd_tree<KdTreeXfL1>("KdTreeXfL1", m);
  def_kd_tree<KdTreeXfL2Squared>("KdTreeXfL2Squared", m);
  def_kd_tree<KdTreeXfLInf>("KdTreeXfLInf", m);
  def_kd_tree<KdTreeXdL1>("KdTreeXdL1", m);
  def_kd_tree<KdTreeXdL2Squared>("KdTreeXdL2Squared", m);
  def_kd_tree<KdTreeXdLInf>("KdTreeXdLInf", m);
}

}  // namespace pyco_tree
