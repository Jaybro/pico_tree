#include "def_kd_tree.hpp"

#include "darray.hpp"
#include "def_core.hpp"
#include "kd_tree.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename PointX>
using KdTreeL1 = KdTree<TraitsX<PointX>, L1<TraitsX<PointX>>>;
template <typename PointX>
using KdTreeL2 = KdTree<TraitsX<PointX>, L2Squared<TraitsX<PointX>>>;

// Dynamic size KdTree
using KdTreeXfL1 = KdTreeL1<PointsXf>;
using KdTreeXfL2 = KdTreeL2<PointsXf>;
using KdTreeXdL1 = KdTreeL1<PointsXd>;
using KdTreeXdL2 = KdTreeL2<PointsXd>;

// Fixed size 2d KdTree
using KdTree2fL1 = KdTreeL1<Points2f>;
using KdTree2fL2 = KdTreeL2<Points2f>;
using KdTree2dL1 = KdTreeL1<Points2d>;
using KdTree2dL2 = KdTreeL2<Points2d>;

// Fixed size 3d KdTree
using KdTree3fL1 = KdTreeL1<Points3f>;
using KdTree3fL2 = KdTreeL2<Points3f>;
using KdTree3dL1 = KdTreeL1<Points3d>;
using KdTree3dL2 = KdTreeL2<Points3d>;

template <typename KdTree>
void DefKdTree(std::string const& name, py::module* m) {
  using Index = typename KdTree::IndexType;
  using Scalar = typename KdTree::ScalarType;
  using Metric = typename KdTree::MetricType;
  using Neighbor = typename KdTree::NeighborType;
  using Neighborhoods = DArray;

  py::class_<KdTree>(*m, name.c_str(), py::buffer_protocol())
      .def(
          py::init<py::array_t<Scalar, 0>, Index>(),
          // We keep the input points alive until the KdTree is gone.
          // Nurse: 1 = Implicit first argument: KdTree.
          // Patient: 2 = Second function argument: NumPy array.
          py::keep_alive<1, 2>(),
          py::arg("pts"),
          py::arg("max_leaf_size"),
          "Create a new KdTree from given points and maximum leaf size.")
      // The KdTree can be used as a read-only buffer. It helps to check how the
      // data is interpreted.
      .def_buffer([](KdTree& t) -> py::buffer_info {
        py::ssize_t const sdim = static_cast<py::ssize_t>(t.sdim());
        py::ssize_t const npts = static_cast<py::ssize_t>(t.npts());
        py::ssize_t const inner_stride = sizeof(Scalar);
        py::ssize_t const outer_stride = inner_stride * sdim;

        // There doesn't appear to be a true read only-interface. Hence the rare
        // const_cast.
        return py::buffer_info(
            const_cast<void*>(static_cast<void const*>(t.data())),
            // Item size:
            inner_stride,
            py::format_descriptor<Scalar>::format(),
            // Array/Tensor dimensions:
            2,
            // Shape:
            t.row_major() ? std::vector<py::ssize_t>{npts, sdim}
                          : std::vector<py::ssize_t>{sdim, npts},
            // Strides:
            t.row_major()
                ? std::vector<py::ssize_t>{outer_stride, inner_stride}
                : std::vector<py::ssize_t>{inner_stride, outer_stride},
            // Read only (at least from the KdTree side of things)
            true);
      })
      .def(
          "__repr__",
          [](KdTree const& t) {
            return "KdTree(metric=" + StringTraits<Metric>::String() +
                   ", dtype=" + StringTraits<Scalar>::String() +
                   ", sdim=" + std::to_string(t.sdim()) +
                   ", npts=" + std::to_string(t.npts()) + ")";
          })
      .def_property_readonly(
          "dtype_index",
          [](KdTree const&) { return py::dtype::of<Index>(); },
          "Get the dtype of indices returned by this tree.")
      .def_property_readonly(
          "dtype_scalar",
          [](KdTree const&) { return py::dtype::of<Scalar>(); },
          "Get the dtype of scalars returned by this tree.")
      .def_property_readonly(
          "dtype_neighbor",
          [](KdTree const&) { return py::dtype::of<Neighbor>(); },
          "Get the dtype of neighbors returned by this tree.")
      .def_property_readonly(
          "sdim", &KdTree::sdim, "Get the spatial dimension of the KdTree.")
      .def_property_readonly(
          "npts", &KdTree::npts, "Get the number of points of the KdTree.")
      // Because of the ambiguity that arises from function overloading, we have
      // to show which function we mean by using a static_cast.
      .def(
          "metric",
          static_cast<Scalar (KdTree::*)(Scalar const) const>(&KdTree::metric),
          py::arg("scalar"),
          "Return a scalar with the metric applied.")
      .def(
          "search_knn",
          static_cast<void (KdTree::*)(
              py::array_t<Scalar, 0> const,
              Index const,
              py::array_t<Neighbor, 0>) const>(&KdTree::SearchKnn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("nns").noconvert().none(false),
          "Search the k nearest neighbors for each of the input points and "
          "store the result in the specified output array. The output will be "
          "resized when its shape is not (npts, k). If resized, its memory "
          "layout will be the same as that of the input.")
      .def(
          "search_knn",
          static_cast<py::array_t<Neighbor, 0> (KdTree::*)(
              py::array_t<Scalar, 0> const, Index const) const>(
              &KdTree::SearchKnn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          "Search the k nearest neighbors for each of the input points. The "
          "memory layout of the output will be the same as that of the input.")
      .def(
          "search_aknn",
          static_cast<void (KdTree::*)(
              py::array_t<Scalar, 0> const,
              Index const,
              Scalar const,
              py::array_t<Neighbor, 0>) const>(&KdTree::SearchAknn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("e").none(false),
          py::arg("nns").noconvert().none(false),
          "Search the k approximate nearest neighbors for each of the input "
          "points and store the result in the specified output array. The "
          "output will be resized when its shape is not (npts, k). If resized, "
          "its memory layout will be the same as that of the input.")
      .def(
          "search_aknn",
          static_cast<py::array_t<Neighbor, 0> (KdTree::*)(
              py::array_t<Scalar, 0> const, Index const, Scalar const) const>(
              &KdTree::SearchAknn),
          py::arg("pts").noconvert().none(false),
          py::arg("k").none(false),
          py::arg("e").none(false),
          "Search the k approximate nearest neighbors for each of the input "
          "points. The memory layout of the output will be the same as that of "
          "the input.")
      .def(
          "search_radius",
          static_cast<void (KdTree::*)(
              py::array_t<Scalar, 0> const,
              Scalar const,
              Neighborhoods*,
              bool const) const>(&KdTree::SearchRadius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("nns").noconvert().none(false),
          py::arg("sort").none(false) = false,
          "Search for all neighbors within a radius of each of the input "
          "points and store the result in the specified output.")
      .def(
          "search_radius",
          static_cast<Neighborhoods (KdTree::*)(
              py::array_t<Scalar, 0> const, Scalar const, bool const) const>(
              &KdTree::SearchRadius),
          py::arg("pts").noconvert().none(false),
          py::arg("radius").none(false),
          py::arg("sort").none(false) = false,
          "Search for all neighbors within a radius of each of the input "
          "points.")
      .def(
          "search_box",
          static_cast<void (KdTree::*)(
              py::array_t<Scalar, 0> const,
              py::array_t<Scalar, 0> const,
              Neighborhoods*) const>(&KdTree::SearchBox),
          py::arg("min").noconvert().none(false),
          py::arg("max").noconvert().none(false),
          py::arg("box").noconvert().none(false),
          "Search for all points within each of the axis aligned input boxes "
          "and store the result in the specified output.")
      .def(
          "search_box",
          static_cast<Neighborhoods (KdTree::*)(
              py::array_t<Scalar, 0> const, py::array_t<Scalar, 0> const)
                          const>(&KdTree::SearchBox),
          py::arg("min").noconvert().none(false),
          py::arg("max").noconvert().none(false),
          "Search for all points within each of the axis aligned input boxes.");
}

void DefKdTree(py::module* m) {
  // Dynamic size KdTree
  DefKdTree<KdTreeXfL1>("KdTreeXfL1", m);
  DefKdTree<KdTreeXfL2>("KdTreeXfL2", m);
  DefKdTree<KdTreeXdL1>("KdTreeXdL1", m);
  DefKdTree<KdTreeXdL2>("KdTreeXdL2", m);

  // Fixed size 2d KdTree
  DefKdTree<KdTree2fL1>("KdTree2fL1", m);
  DefKdTree<KdTree2fL2>("KdTree2fL2", m);
  DefKdTree<KdTree2dL1>("KdTree2dL1", m);
  DefKdTree<KdTree2dL2>("KdTree2dL2", m);

  // Fixed size 3d KdTree
  DefKdTree<KdTree3fL1>("KdTree3fL1", m);
  DefKdTree<KdTree3fL2>("KdTree3fL2", m);
  DefKdTree<KdTree3dL1>("KdTree3dL1", m);
  DefKdTree<KdTree3dL2>("KdTree3dL2", m);
}

}  // namespace pyco_tree
