#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/kd_tree.hpp>
#include <thread>

#include "darray.hpp"
#include "pyco_adaptor.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief pico_tree::KdTree with some added convenience functions to be mapped
//! on the Python side of things.
//! \see pico_tree::KdTree
template <typename Points, typename Metric>
class KdTree : public pico_tree::KdTree<
                   typename Points::IndexType,
                   typename Points::ScalarType,
                   Points::Dim,
                   Points,
                   Metric> {
 private:
  static constexpr int kChunkSize = 256;

  using Index = typename Points::IndexType;
  using Scalar = typename Points::ScalarType;
  using Base = pico_tree::KdTree<Index, Scalar, Points::Dim, Points, Metric>;

 public:
  using Base::Dim;
  using Base::metric;
  using Base::points;
  using Base::SearchAknn;
  using Base::SearchKnn;
  using Base::SearchRadius;
  using typename Base::IndexType;
  using typename Base::MetricType;
  using typename Base::NeighborType;
  using typename Base::PointsType;
  using typename Base::ScalarType;

 public:
  inline KdTree(Points points, Index max_leaf_size)
      : Base(std::move(points), max_leaf_size) {}

  void SearchKnns(
      py::array_t<Scalar, 0> const pts,
      Index const k,
      py::array_t<NeighborType, 0> nns) const {
    PycoAdaptor<Index, Scalar, Dim> query(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(static, kChunkSize)
    for (Index i = 0; i < query.npts(); ++i) {
      SearchKnn(query(i), output + i * k, output + (i + 1) * k);
    }
  }

  py::array_t<NeighborType, 0> SearchKnns(
      py::array_t<Scalar, 0> const pts, Index const k) const {
    py::array_t<NeighborType, 0> nns;
    SearchKnns(pts, k, nns);
    return nns;
  }

  void SearchAknns(
      py::array_t<Scalar, 0> const pts,
      Index const k,
      Scalar const e,
      py::array_t<NeighborType, 0> nns) const {
    PycoAdaptor<Index, Scalar, Dim> query(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(static, kChunkSize)
    for (Index i = 0; i < query.npts(); ++i) {
      SearchAknn(query(i), e, output + i * k, output + (i + 1) * k);
    }
  }

  py::array_t<NeighborType, 0> SearchAknns(
      py::array_t<Scalar, 0> const pts, Index const k, Scalar const e) const {
    py::array_t<NeighborType, 0> nns;
    SearchAknns(pts, k, e, nns);
    return nns;
  }

  void SearchRadius(
      py::array_t<Scalar, 0> const pts,
      Scalar const radius,
      DArray* nns,
      bool const sort) const {
    PycoAdaptor<Index, Scalar, Dim> query(pts);

    auto& nns_data = nns->data<NeighborType>();
    nns_data.resize(query.npts());

#pragma omp parallel for schedule(static, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (Index i = 0; i < query.npts(); ++i) {
      SearchRadius(query(i), radius, &nns_data[i], sort);
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

  inline Scalar const* const data() const { return points().data(); }

  inline int sdim() const { return points().sdim(); }

  inline Index npts() const { return points().npts(); }

  inline bool row_major() const { return points().row_major(); }

  inline Scalar metric(Scalar const v) const { return metric()(v); }

 private:
  void EnsureSize(
      Points const& query,
      Index const k,
      py::array_t<NeighborType, 0> nns) const {
    // This respects the ndim == 1 for k == 1
    if (nns.size() != static_cast<py::ssize_t>(query.npts() * k)) {
      // Resize regardless of the reference count.
      if (k == 1) {
        nns.resize(std::vector<py::ssize_t>{query.npts()}, false);
      } else {
        nns.resize(
            query.row_major() ? std::vector<py::ssize_t>{query.npts(), k}
                              : std::vector<py::ssize_t>{k, query.npts()},
            false);
      }
    }
  }
};

}  // namespace pyco_tree