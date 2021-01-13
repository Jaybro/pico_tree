#include "def_core.hpp"

#include <pybind11/stl_bind.h>

namespace py = pybind11;

// Disable template based conversions of the following types:
PYBIND11_MAKE_OPAQUE(pyco_tree::Neighborhoodf);
PYBIND11_MAKE_OPAQUE(pyco_tree::Neighborhoodd);
PYBIND11_MAKE_OPAQUE(pyco_tree::Neighborhoodsf);
PYBIND11_MAKE_OPAQUE(pyco_tree::Neighborhoodsd);

namespace pyco_tree {

template <typename Neighbor>
void DefNeighbor(std::string const& name, py::module* m) {
  py::class_<Neighbor>(*m, name.c_str())
      .def_readwrite("index", &Neighbor::index)
      .def_readwrite("distance", &Neighbor::distance)
      // Prints the contents of a Neighbor in the same format as that of a
      // struct in a structured NumPy array. That is, an element of:
      // py::array<Neighbor, 0>.
      // TODO The default C++ to string formats a double differently than NumPy
      // does.
      .def(
          "__repr__",
          [](Neighbor const& n) {
            return "(" + std::to_string(n.index) + ", " +
                   std::to_string(n.distance) + ")";
          })
      .def(
          "dtype",
          [](Neighbor const& n) { return py::dtype::of<Neighbor>(); },
          "Return the dtype of a Neighbor.");
}

template <typename Neighborhood>
void DefNeighborhood(std::string const& name, py::module* m) {
  using Neighbor = typename Neighborhood::value_type;

  py::bind_vector<Neighborhood>(*m, name.c_str(), py::module_local(true))
      .def(
          "__repr__",
          [](Neighborhood const& n) {
            return "Neighborhood(dtype=" +
                   StringTraits<typename Neighborhood::value_type::ScalarType>::
                       String() +
                   ", npts=" + std::to_string(n.size()) + ")";
          })
      .def(
          "numpy",
          [](Neighborhood& n) -> py::array_t<Neighbor, 0> {
            // A NumPy array has ownership of its own data when it is created
            // with default arguments. This causes it to copy the data from the
            // input pointer unless we tell it that an other object should own
            // its data. In this case that other object equals py::none().
            // It would have been nice if that could be this/self?
            // In this case it is important that while the array is alive, the
            // Neighborhood is kept alive (see py::keep_alive).
            // NOTE: At the time of writing an undocumented feature.
            return py::array_t<Neighbor, 0>(n.size(), n.data(), py::none());
          },
          // The return value keeps this object alive.
          py::keep_alive<0, 1>(),
          "Return a numpy array view of this neighborhood.");
}

template <typename Neighborhoods>
void DefNeighborhoods(std::string const& name, py::module* m) {
  py::bind_vector<Neighborhoods>(*m, name.c_str(), py::module_local(true))
      .def("__repr__", [](Neighborhoods const& n) {
        return "Neighborhoods(dtype=" +
               StringTraits<typename Neighborhoods::value_type::value_type::
                                ScalarType>::String() +
               ", nhds=" + std::to_string(n.size()) + ")";
      });
}

void DefCore(py::module* m) {
  // Registered neighbor types.
  DefNeighbor<Neighborf>("Neighborf", m);
  DefNeighbor<Neighbord>("Neighbord", m);

  // Registered vectors.
  DefNeighborhood<Neighborhoodf>("Neighborhoodf", m);
  DefNeighborhood<Neighborhoodd>("Neighborhoodd", m);

  DefNeighborhoods<Neighborhoodsf>("Neighborhoodsf", m);
  DefNeighborhoods<Neighborhoodsd>("Neighborhoodsd", m);
}

}  // namespace pyco_tree
