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
template <typename Space_, typename Metric_>
class KdTree : public pico_tree::KdTree<Space_, Metric_> {
 private:
  static constexpr int kChunkSize = 128;

  using Base = pico_tree::KdTree<Space_, Metric_>;
  using Space = Space_;
  // TODO Remove when MSVC++ has default support for OpenMP 3.0+.
  using SSize = std::ptrdiff_t;

 public:
  using Base::Dim;
  using Base::metric;
  using Base::points;
  using typename Base::IndexType;
  using typename Base::MetricType;
  using typename Base::NeighborType;
  using typename Base::ScalarType;
  using typename Base::SizeType;
  using typename Base::SpaceType;

 public:
  inline KdTree(py::array_t<ScalarType, 0> pts, SizeType max_leaf_size)
      : Base(MakeMap<Dim>(pts), pico_tree::max_leaf_size_t(max_leaf_size)) {}

  void SearchKnn(
      py::array_t<ScalarType, 0> const pts,
      SizeType const k,
      py::array_t<NeighborType, 0> nns) const {
    auto query = MakeMap<Dim>(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    for (SSize i = 0; i < static_cast<SSize>(query.size()); ++i) {
      auto index = static_cast<SizeType>(i * k);
      Base::SearchKnn(query[i], output + index, output + index + k);
    }
  }

  py::array_t<NeighborType, 0> SearchKnn(
      py::array_t<ScalarType, 0> const pts, SizeType const k) const {
    py::array_t<NeighborType, 0> nns;
    SearchKnn(pts, k, nns);
    return nns;
  }

  void SearchKnn(
      py::array_t<ScalarType, 0> const pts,
      SizeType const k,
      ScalarType const e,
      py::array_t<NeighborType, 0> nns) const {
    auto query = MakeMap<Dim>(pts);
    EnsureSize(query, k, nns);
    auto output = static_cast<NeighborType*>(nns.mutable_data());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    for (SSize i = 0; i < static_cast<SSize>(query.size()); ++i) {
      auto index = static_cast<SizeType>(i * k);
      Base::SearchKnn(query[i], e, output + index, output + index + k);
    }
  }

  py::array_t<NeighborType, 0> SearchKnn(
      py::array_t<ScalarType, 0> const pts,
      SizeType const k,
      ScalarType const e) const {
    py::array_t<NeighborType, 0> nns;
    SearchKnn(pts, k, e, nns);
    return nns;
  }

  void SearchRadius(
      py::array_t<ScalarType, 0> const pts,
      ScalarType const radius,
      DArray& nns,
      bool const sort) const {
    auto query = MakeMap<Dim>(pts);

    auto& nns_data = nns.data<NeighborType>();
    nns_data.resize(query.size());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (SSize i = 0; i < static_cast<SSize>(query.size()); ++i) {
      Base::SearchRadius(query[i], radius, nns_data[i], sort);
    }
  }

  DArray SearchRadius(
      py::array_t<ScalarType, 0> const pts,
      ScalarType const radius,
      bool const sort) const {
    DArray nns = DArray(std::vector<std::vector<NeighborType>>());
    SearchRadius(pts, radius, nns, sort);
    return nns;
  }

  void SearchRadius(
      py::array_t<ScalarType, 0> const pts,
      ScalarType const radius,
      ScalarType const e,
      DArray& nns,
      bool const sort) const {
    auto query = MakeMap<Dim>(pts);

    auto& nns_data = nns.data<NeighborType>();
    nns_data.resize(query.size());

#pragma omp parallel for schedule(dynamic, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (SSize i = 0; i < static_cast<SSize>(query.size()); ++i) {
      Base::SearchRadius(query[i], radius, e, nns_data[i], sort);
    }
  }

  DArray SearchRadius(
      py::array_t<ScalarType, 0> const pts,
      ScalarType const radius,
      ScalarType const e,
      bool const sort) const {
    DArray nns = DArray(std::vector<std::vector<NeighborType>>());
    SearchRadius(pts, radius, e, nns, sort);
    return nns;
  }

  void SearchBox(
      py::array_t<ScalarType, 0> const boxes, DArray& indices) const {
    auto query = MakeMap<Dim>(boxes);

    if (query.size() % 2 != 0) {
      throw std::invalid_argument("Query min and max don't have equal size.");
    }

    std::size_t box_count = query.size() / std::size_t(2);
    auto& indices_data = indices.data<IndexType>();
    indices_data.resize(box_count);

#pragma omp parallel for schedule(dynamic, kChunkSize)
    // TODO Reduce the vector resize overhead
    for (SSize i = 0; i < static_cast<SSize>(box_count); ++i) {
      auto index = static_cast<SizeType>(i * 2);
      Base::SearchBox(query[index + 0], query[index + 1], indices_data[i]);
    }
  }

  DArray SearchBox(py::array_t<ScalarType, 0> const boxes) const {
    DArray indices = DArray(std::vector<std::vector<IndexType>>());
    SearchBox(boxes, indices);
    return indices;
  }

  inline ScalarType const* data() const { return points().data(); }

  inline int sdim() const { return static_cast<int>(points().sdim()); }

  inline SizeType npts() const { return points().size(); }

  inline bool row_major() const { return points().row_major(); }

  inline ScalarType metric(ScalarType v) const { return metric()(v); }

 private:
  void EnsureSize(
      Space const& query,
      SizeType const k,
      py::array_t<NeighborType, 0> nns) const {
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
