#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/kd_tree.hpp>
#include <thread>

#include "darray.hpp"
#include "py_array_map.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief pico_tree::kd_tree with some added convenience functions to be mapped
//! on the Python side of things.
//! \see pico_tree::kd_tree
template <typename Space_, typename Metric_>
class kd_tree : public pico_tree::kd_tree<Space_, Metric_> {
 private:
  static constexpr int chunk_size = 128;

  using base = pico_tree::kd_tree<Space_, Metric_>;
  // TODO Remove when MSVC++ has default support for OpenMP 3.0+.
  using ssize_type = std::ptrdiff_t;

 public:
  using base::dim;
  using base::metric;
  using base::points;
  using typename base::index_type;
  using typename base::metric_type;
  using typename base::neighbor_type;
  using typename base::scalar_type;
  using typename base::size_type;
  using typename base::space_type;

 public:
  inline kd_tree(py::array_t<scalar_type, 0> pts, size_type max_leaf_size)
      : base(make_map<dim>(pts), pico_tree::max_leaf_size_t(max_leaf_size)) {}

  void search_knn(
      py::array_t<scalar_type, 0> const pts,
      size_type const k,
      py::array_t<neighbor_type, 0> nns) const {
    auto query = make_map<dim>(pts);
    ensure_size(query, k, nns);
    auto output = static_cast<neighbor_type*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (ssize_type i = 0; i < static_cast<ssize_type>(query.size()); ++i) {
      auto index = static_cast<size_type>(i * k);
      base::search_knn(query[i], output + index, output + index + k);
    }
  }

  py::array_t<neighbor_type, 0> search_knn(
      py::array_t<scalar_type, 0> const pts, size_type const k) const {
    py::array_t<neighbor_type, 0> nns;
    search_knn(pts, k, nns);
    return nns;
  }

  void search_knn(
      py::array_t<scalar_type, 0> const pts,
      size_type const k,
      scalar_type const e,
      py::array_t<neighbor_type, 0> nns) const {
    auto query = make_map<dim>(pts);
    ensure_size(query, k, nns);
    auto output = static_cast<neighbor_type*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (ssize_type i = 0; i < static_cast<ssize_type>(query.size()); ++i) {
      auto index = static_cast<size_type>(i * k);
      base::search_knn(query[i], e, output + index, output + index + k);
    }
  }

  py::array_t<neighbor_type, 0> search_knn(
      py::array_t<scalar_type, 0> const pts,
      size_type const k,
      scalar_type const e) const {
    py::array_t<neighbor_type, 0> nns;
    search_knn(pts, k, e, nns);
    return nns;
  }

  void search_radius(
      py::array_t<scalar_type, 0> const pts,
      scalar_type const radius,
      darray& nns,
      bool const sort) const {
    auto query = make_map<dim>(pts);

    auto& nns_data = nns.data<neighbor_type>();
    nns_data.resize(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type i = 0; i < static_cast<ssize_type>(query.size()); ++i) {
      base::search_radius(query[i], radius, nns_data[i], sort);
    }
  }

  darray search_radius(
      py::array_t<scalar_type, 0> const pts,
      scalar_type const radius,
      bool const sort) const {
    darray nns = darray(std::vector<std::vector<neighbor_type>>());
    search_radius(pts, radius, nns, sort);
    return nns;
  }

  void search_radius(
      py::array_t<scalar_type, 0> const pts,
      scalar_type const radius,
      scalar_type const e,
      darray& nns,
      bool const sort) const {
    auto query = make_map<dim>(pts);

    auto& nns_data = nns.data<neighbor_type>();
    nns_data.resize(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type i = 0; i < static_cast<ssize_type>(query.size()); ++i) {
      base::search_radius(query[i], radius, e, nns_data[i], sort);
    }
  }

  darray search_radius(
      py::array_t<scalar_type, 0> const pts,
      scalar_type const radius,
      scalar_type const e,
      bool const sort) const {
    darray nns = darray(std::vector<std::vector<neighbor_type>>());
    search_radius(pts, radius, e, nns, sort);
    return nns;
  }

  void search_box(
      py::array_t<scalar_type, 0> const boxes, darray& indices) const {
    auto query = make_map<dim>(boxes);

    if (query.size() % 2 != 0) {
      throw std::invalid_argument("Query min and max don't have equal size.");
    }

    std::size_t box_count = query.size() / std::size_t(2);
    auto& indices_data = indices.data<index_type>();
    indices_data.resize(box_count);

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type i = 0; i < static_cast<ssize_type>(box_count); ++i) {
      auto index = static_cast<size_type>(i * 2);
      base::search_box(query[index + 0], query[index + 1], indices_data[i]);
    }
  }

  darray search_box(py::array_t<scalar_type, 0> const boxes) const {
    darray indices = darray(std::vector<std::vector<index_type>>());
    search_box(boxes, indices);
    return indices;
  }

  inline scalar_type const* data() const { return points().data(); }

  inline int sdim() const { return static_cast<int>(points().sdim()); }

  inline size_type npts() const { return points().size(); }

  inline bool row_major() const { return points().row_major(); }

  inline scalar_type metric(scalar_type v) const { return metric()(v); }

 private:
  void ensure_size(
      space_type const& query,
      size_type const k,
      py::array_t<neighbor_type, 0> nns) const {
    // This respects the ndim == 1 for k == 1
    if (nns.size() != static_cast<py::ssize_t>(query.size() * k)) {
      // Resize regardless of the reference count.
      py::ssize_t npts = static_cast<py::ssize_t>(query.size());
      if (k == 1) {
        nns.resize(std::vector<py::ssize_t>{npts}, false);
      } else {
        py::ssize_t sk = static_cast<py::ssize_t>(k);
        nns.resize(
            query.row_major() ? std::vector<py::ssize_t>{npts, sk}
                              : std::vector<py::ssize_t>{sk, npts},
            false);
      }
    }
  }
};

}  // namespace pyco_tree
