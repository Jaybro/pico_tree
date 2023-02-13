#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/kd_tree.hpp>
#include <thread>

#include "darray.hpp"
#include "py_array_map.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief pico_tree::KdTree with some added convenience functions to be mapped
//! on the Python side of things.
//! \see pico_tree::KdTree
template <typename Traits, typename Metric>
class KdTree : public pico_tree::KdTree<Traits, Metric> {
 private:
  static constexpr int kChunkSize = 128;

  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  using Base = pico_tree::KdTree<Traits, Metric>;

 public:
  using Base::Dim;
  using Base::metric;
  using Base::points;
  using typename Base::IndexType;
  using typename Base::MetricType;
  using typename Base::NeighborType;
  using typename Base::ScalarType;
  using typename Base::SpaceType;

 public:
  inline KdTree(py::array_t<Scalar, 0> pts, Index max_leaf_size)
      : Base(MakeMap<Dim>(pts), max_leaf_size) {}

  void SearchKnn(
      py::array_t<Scalar, 0> const pts,
      Index const k,
      py::array_t<NeighborType, 0> nns) const {
    auto query = MakeMap<Dim>(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    for (Index i = 0; i < static_cast<Index>(query.size()); ++i) {
      Base::SearchKnn(query[i], output + i * k, output + (i + 1) * k);
    }
  }

  py::array_t<NeighborType, 0> SearchKnn(
      py::array_t<Scalar, 0> const pts, Index const k) const {
    py::array_t<NeighborType, 0> nns;
    SearchKnn(pts, k, nns);
    return nns;
  }

  void SearchAknn(
      py::array_t<Scalar, 0> const pts,
      Index const k,
      Scalar const e,
      py::array_t<NeighborType, 0> nns) const {
    auto query = MakeMap<Dim>(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    for (Index i = 0; i < static_cast<Index>(query.size()); ++i) {
      Base::SearchAknn(query[i], e, output + i * k, output + (i + 1) * k);
    }
  }

  py::array_t<NeighborType, 0> SearchAknn(
      py::array_t<Scalar, 0> const pts, Index const k, Scalar const e) const {
    py::array_t<NeighborType, 0> nns;
    SearchAknn(pts, k, e, nns);
    return nns;
  }

  void SearchRadius(
      py::array_t<Scalar, 0> const pts,
      Scalar const radius,
      DArray* nns,
      bool const sort) const {
    auto query = MakeMap<Dim>(pts);

    auto& nns_data = nns->data<NeighborType>();
    nns_data.resize(query.size());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (Index i = 0; i < static_cast<Index>(query.size()); ++i) {
      Base::SearchRadius(query[i], radius, &nns_data[i], sort);
    }
  }

  DArray SearchRadius(
      py::array_t<Scalar, 0> const pts,
      Scalar const radius,
      bool const sort) const {
    DArray nns = DArray(std::vector<std::vector<NeighborType>>());
    SearchRadius(pts, radius, &nns, sort);
    return nns;
  }

  void SearchBox(
      py::array_t<Scalar, 0> const min,
      py::array_t<Scalar, 0> const max,
      DArray* box) const {
    auto query_min = MakeMap<Dim>(min);
    auto query_max = MakeMap<Dim>(max);

    if (query_min.size() != query_max.size()) {
      throw std::invalid_argument("Query min and max don't have equal size.");
    }

    auto& box_data = box->data<Index>();
    box_data.resize(query_min.size());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (Index i = 0; i < static_cast<Index>(query_min.size()); ++i) {
      Base::SearchBox(query_min[i], query_max[i], &box_data[i]);
    }
  }

  DArray SearchBox(
      py::array_t<Scalar, 0> const min,
      py::array_t<Scalar, 0> const max) const {
    DArray box = DArray(std::vector<std::vector<Index>>());
    SearchBox(min, max, &box);
    return box;
  }

  inline Scalar const* data() const { return points().data(); }

  inline int sdim() const { return static_cast<int>(points().sdim()); }

  inline Index npts() const { return static_cast<Index>(points().size()); }

  inline bool row_major() const { return points().row_major(); }

  inline Scalar metric(Scalar const v) const { return metric()(v); }

 private:
  void EnsureSize(
      Space const& query,
      Index const k,
      py::array_t<NeighborType, 0> nns) const {
    // This respects the ndim == 1 for k == 1
    if (nns.size() != static_cast<py::ssize_t>(query.size() * k)) {
      // Resize regardless of the reference count.
      py::ssize_t npts = static_cast<py::ssize_t>(query.size());
      if (k == 1) {
        nns.resize(std::vector<py::ssize_t>{npts}, false);
      } else {
        nns.resize(
            query.row_major() ? std::vector<py::ssize_t>{npts, k}
                              : std::vector<py::ssize_t>{k, npts},
            false);
      }
    }
  }
};

}  // namespace pyco_tree
