#pragma once

#include <pybind11/numpy.h>

#include <fstream>
#include <pico_tree/kd_tree.hpp>
#include <stdexcept>
#include <thread>

#include "darray.hpp"
#include "py_array_map.hpp"

namespace py = pybind11;

namespace pyco_tree {

enum class metric_t : int { l1, l2_squared, linf };

namespace internal {

class kd_tree_base {
 public:
  virtual void search_knn(
      py::array const pts, py::int_ const k, py::array nns) const = 0;

  virtual py::array search_knn(py::array const pts, py::int_ const k) const = 0;

  virtual void search_knn(
      py::array const pts,
      py::int_ const k,
      py::float_ const e,
      py::array nns) const = 0;

  virtual py::array search_knn(
      py::array const pts, py::int_ const k, py::float_ const e) const = 0;

  virtual void search_radius(
      py::array const pts,
      py::float_ const radius,
      darray& nns,
      bool const sort) const = 0;

  virtual darray search_radius(
      py::array const pts, py::float_ const radius, bool const sort) const = 0;

  virtual void search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      darray& nns,
      bool const sort) const = 0;

  virtual darray search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      bool const sort) const = 0;

  virtual void search_box(py::array const boxes, darray& indices) const = 0;

  virtual darray search_box(py::array const boxes) const = 0;

  virtual py::float_ metric(py::float_ const v) const = 0;

  virtual void save(std::iostream& stream) const = 0;

  virtual void const* data() const = 0;

  virtual py::int_ sdim() const = 0;

  virtual py::int_ npts() const = 0;

  virtual py::buffer_info request() const = 0;

  virtual std::string metric_string() const = 0;

  virtual std::string repr() const = 0;

  virtual py::dtype dtype_index() const = 0;

  virtual py::dtype dtype_scalar() const = 0;

  virtual py::dtype dtype_neighbor() const = 0;

  virtual ~kd_tree_base() = default;
};

//! \brief pico_tree::kd_tree with some added convenience functions to be mapped
//! on the Python side of things.
//! \see pico_tree::kd_tree
template <typename Space_, typename Metric_>
class kd_tree_impl final : public kd_tree_base {
 private:
  static constexpr int chunk_size = 128;

  using kd_tree_type = pico_tree::kd_tree<Space_, Metric_>;
  // TODO Remove when MSVC++ has default support for OpenMP 3.0+.
  using ssize_type = std::ptrdiff_t;

 public:
  using space_type = typename kd_tree_type::space_type;
  using scalar_type = typename kd_tree_type::scalar_type;
  auto static constexpr dim = kd_tree_type::dim;
  using index_type = typename kd_tree_type::index_type;
  using metric_type = typename kd_tree_type::metric_type;
  using neighbor_type = typename kd_tree_type::neighbor_type;
  using size_type = typename kd_tree_type::size_type;

  inline kd_tree_impl(py::array pts, size_type max_leaf_size)
      : tree_(
            make_map<dim, scalar_type>(pts),
            pico_tree::max_leaf_size_t(max_leaf_size)) {}

  inline kd_tree_impl(py::array pts, std::iostream& stream)
      : tree_(kd_tree_type::load(make_map<dim, scalar_type>(pts), stream)) {}

  void search_knn(py::array const pts, py::int_ const k_nearest, py::array nns)
      const override {
    size_type k = size_type(k_nearest);
    throw_if_not_scalar_dtype(pts);
    throw_if_not_neighbor_dtype(nns);

    auto query = make_map<dim, scalar_type>(pts);
    ensure_size(query, k, nns);
    auto output = static_cast<neighbor_type*>(nns.mutable_data());
    auto const point_count = static_cast<ssize_type>(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (ssize_type si_point_in = 0; si_point_in < point_count; ++si_point_in) {
      size_type ui_point_in = static_cast<size_type>(si_point_in);
      size_type ui_point_out = ui_point_in * k;
      tree_.search_knn(
          query[ui_point_in], output + ui_point_out, output + ui_point_out + k);
    }
  }

  py::array search_knn(
      py::array const pts, py::int_ const k_nearest) const override {
    py::array_t<neighbor_type, 0> nns;
    search_knn(pts, k_nearest, nns);
    return nns;
  }

  void search_knn(
      py::array const pts,
      py::int_ const k_nearest,
      py::float_ const e,
      py::array nns) const override {
    size_type k = size_type(k_nearest);
    throw_if_not_scalar_dtype(pts);
    throw_if_not_neighbor_dtype(nns);

    auto query = make_map<dim, scalar_type>(pts);
    ensure_size(query, k, nns);
    auto output = static_cast<neighbor_type*>(nns.mutable_data());
    auto const point_count = static_cast<ssize_type>(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (ssize_type si_point_in = 0; si_point_in < point_count; ++si_point_in) {
      size_type ui_point_in = static_cast<size_type>(si_point_in);
      size_type ui_point_out = ui_point_in * k;
      tree_.search_knn(
          query[ui_point_in],
          e,
          output + ui_point_out,
          output + ui_point_out + k);
    }
  }

  py::array search_knn(
      py::array const pts,
      py::int_ const k_nearest,
      py::float_ const e) const override {
    py::array_t<neighbor_type, 0> nns;
    search_knn(pts, k_nearest, e, nns);
    return nns;
  }

  void search_radius(
      py::array const pts,
      py::float_ const radius,
      darray& nns,
      bool const sort) const override {
    throw_if_not_scalar_dtype(pts);
    throw_if_not_neighbor_dtype(nns);

    auto query = make_map<dim, scalar_type>(pts);

    auto& nns_data = nns.data<neighbor_type>();
    nns_data.resize(query.size());
    auto const point_count = static_cast<ssize_type>(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type si_point_in = 0; si_point_in < point_count; ++si_point_in) {
      size_type ui_point_in = static_cast<size_type>(si_point_in);
      tree_.search_radius(
          query[ui_point_in], radius, nns_data[ui_point_in], sort);
    }
  }

  darray search_radius(
      py::array const pts,
      py::float_ const radius,
      bool const sort) const override {
    darray nns = darray(std::vector<std::vector<neighbor_type>>());
    search_radius(pts, radius, nns, sort);
    return nns;
  }

  void search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      darray& nns,
      bool const sort) const override {
    throw_if_not_scalar_dtype(pts);
    throw_if_not_neighbor_dtype(nns);

    auto query = make_map<dim, scalar_type>(pts);

    auto& nns_data = nns.data<neighbor_type>();
    nns_data.resize(query.size());
    auto const point_count = static_cast<ssize_type>(query.size());

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type si_point_in = 0; si_point_in < point_count; ++si_point_in) {
      size_type ui_point_in = static_cast<size_type>(si_point_in);
      tree_.search_radius(
          query[ui_point_in], radius, e, nns_data[ui_point_in], sort);
    }
  }

  darray search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      bool const sort) const override {
    darray nns = darray(std::vector<std::vector<neighbor_type>>());
    search_radius(pts, radius, e, nns, sort);
    return nns;
  }

  void search_box(py::array const boxes, darray& indices) const override {
    throw_if_not_scalar_dtype(boxes);
    throw_if_not_index_dtype(indices);

    auto query = make_map<dim, scalar_type>(boxes);

    if (query.size() % 2 != 0) {
      throw std::invalid_argument("query min and max don't have equal size");
    }

    ssize_type box_count =
        static_cast<ssize_type>(query.size()) / ssize_type(2);
    auto& indices_data = indices.data<index_type>();
    indices_data.resize(static_cast<size_type>(box_count));

#pragma omp parallel for schedule(dynamic, chunk_size)
    // TODO Reduce the vector resize overhead
    for (ssize_type si_box = 0; si_box < box_count; ++si_box) {
      size_type ui_box = static_cast<size_type>(si_box);
      size_type ui_box_min = ui_box * 2;
      tree_.search_box(
          query[ui_box_min + 0], query[ui_box_min + 1], indices_data[ui_box]);
    }
  }

  darray search_box(py::array const boxes) const override {
    darray indices = darray(std::vector<std::vector<index_type>>());
    search_box(boxes, indices);
    return indices;
  }

  inline py::float_ metric(py::float_ const v) const override {
    return tree_.metric()(scalar_type(v));
  }

  inline void save(std::iostream& stream) const override {
    kd_tree_type::save(tree_, stream);
  }

  inline void const* data() const override {
    return static_cast<void const*>(tree_.space().data());
  }

  inline py::int_ sdim() const override { return tree_.space().sdim(); }

  inline py::int_ npts() const override { return tree_.space().size(); }

  py::buffer_info request() const override {
    py::ssize_t const sdim = static_cast<py::ssize_t>(this->sdim());
    py::ssize_t const npts = static_cast<py::ssize_t>(this->npts());
    py::ssize_t const inner_stride = sizeof(scalar_type);
    py::ssize_t const outer_stride = inner_stride * sdim;

    // There doesn't appear to be a true read only-interface. Hence the rare
    // const_cast.
    return py::buffer_info(
        const_cast<void*>(data()),
        // Item size:
        inner_stride,
        py::format_descriptor<scalar_type>::format(),
        // Array/Tensor dimensions:
        2,
        // Shape:
        row_major() ? std::vector<py::ssize_t>{npts, sdim}
                    : std::vector<py::ssize_t>{sdim, npts},
        // Strides:
        row_major() ? std::vector<py::ssize_t>{outer_stride, inner_stride}
                    : std::vector<py::ssize_t>{inner_stride, outer_stride},
        // Read only (at least from the kd_tree side of things)
        true);
  }

  std::string metric_string() const override {
    return string_traits<metric_type>::type_string();
  }

  std::string repr() const override {
    return "KdTree(metric=" + metric_string() +
           ", dtype=" + string_traits<scalar_type>::type_string() +
           ", sdim=" + std::to_string(tree_.space().sdim()) +
           ", npts=" + std::to_string(tree_.space().size()) + ")";
  }

  py::dtype dtype_index() const override { return py::dtype::of<index_type>(); }

  py::dtype dtype_scalar() const override {
    return py::dtype::of<scalar_type>();
  }

  py::dtype dtype_neighbor() const override {
    return py::dtype::of<neighbor_type>();
  }

 private:
  template <typename T_>
  void throw_if_not_scalar_dtype(T_ const& data) const {
    if (data.dtype().not_equal(dtype_scalar())) {
      throw std::invalid_argument("unexpected dtype_scalar for data");
    }
  }

  template <typename T_>
  void throw_if_not_neighbor_dtype(T_ const& data) const {
    if (data.dtype().not_equal(dtype_neighbor())) {
      throw std::invalid_argument("unexpected dtype_neighbor for data");
    }
  }

  template <typename T_>
  void throw_if_not_index_dtype(T_ const& data) const {
    if (data.dtype().not_equal(dtype_index())) {
      throw std::invalid_argument("unexpected dtype_index for data");
    }
  }

  inline bool row_major() const { return tree_.space().row_major(); }

  void ensure_size(
      space_type const& query, size_type const k, py::array nns) const {
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

  kd_tree_type tree_;
};

template <typename Scalar_, size_t Dim_, typename Arg_>
std::unique_ptr<kd_tree_base> make_kd_tree_from_metric(
    py::array space, metric_t metric, Arg_&& arg) {
  using space_type = py_array_map<Scalar_, Dim_>;

  switch (metric) {
    case metric_t::l1:
      return std::make_unique<kd_tree_impl<space_type, pico_tree::metric_l1>>(
          std::move(space), std::forward<Arg_>(arg));
    case metric_t::l2_squared:
      return std::make_unique<
          kd_tree_impl<space_type, pico_tree::metric_l2_squared>>(
          std::move(space), std::forward<Arg_>(arg));
    case metric_t::linf:
      return std::make_unique<kd_tree_impl<space_type, pico_tree::metric_linf>>(
          std::move(space), std::forward<Arg_>(arg));
  }

  // A throw for what should be unreachable code.
  throw std::invalid_argument("unexpected metric");
}

template <typename Scalar_, typename Arg_>
std::unique_ptr<kd_tree_base> make_kd_tree_from_dim(
    py::array space, metric_t metric, Arg_&& arg) {
  array_layout layout(space);
  // Create a kd_tree with known compile time dimension for the two most common
  // spatial dimension cases. This makes it more likely that the compiler will
  // use SIMD instructions.
  switch (layout.inner_stride()) {
    case 2:
      return make_kd_tree_from_metric<Scalar_, 2>(
          std::move(space), metric, std::forward<Arg_>(arg));
    case 3:
      return make_kd_tree_from_metric<Scalar_, 3>(
          std::move(space), metric, std::forward<Arg_>(arg));
    default:
      return make_kd_tree_from_metric<Scalar_, pico_tree::dynamic_size>(
          std::move(space), metric, std::forward<Arg_>(arg));
  }
}

template <typename Arg_>
std::unique_ptr<kd_tree_base> make_kd_tree_from_scalar(
    py::array space, metric_t metric, Arg_&& arg) {
  if (space.dtype().equal(py::dtype::of<float>())) {
    return make_kd_tree_from_dim<float>(
        std::move(space), metric, std::forward<Arg_>(arg));
  } else if (space.dtype().equal(py::dtype::of<double>())) {
    return make_kd_tree_from_dim<double>(
        std::move(space), metric, std::forward<Arg_>(arg));
  } else {
    throw std::invalid_argument("unexpected space dtype");
  }
}

template <typename Arg_>
std::unique_ptr<kd_tree_base> make_kd_tree(
    py::array space, metric_t metric, Arg_&& arg) {
  return make_kd_tree_from_scalar(
      std::move(space), metric, std::forward<Arg_>(arg));
}

}  // namespace internal

class kd_tree {
 public:
  using index_type = int;
  using size_type = pico_tree::size_t;

  kd_tree(py::array space, metric_t metric, size_type max_leaf_size)
      : impl_(internal::make_kd_tree(std::move(space), metric, max_leaf_size)) {
  }

  kd_tree(py::array space, metric_t metric, std::iostream& stream)
      : impl_(internal::make_kd_tree(std::move(space), metric, stream)) {}

  inline void search_knn(
      py::array const pts, py::int_ const k, py::array nns) const {
    impl_->search_knn(pts, k, nns);
  }

  inline py::array search_knn(py::array const pts, py::int_ const k) const {
    return impl_->search_knn(pts, k);
  }

  inline void search_knn(
      py::array const pts,
      py::int_ const k,
      py::float_ const e,
      py::array nns) const {
    impl_->search_knn(pts, k, e, nns);
  }

  inline py::array search_knn(
      py::array const pts, py::int_ const k, py::float_ const e) const {
    return impl_->search_knn(pts, k, e);
  }

  inline void search_radius(
      py::array const pts,
      py::float_ const radius,
      darray& nns,
      bool const sort) const {
    impl_->search_radius(pts, radius, nns, sort);
  }

  inline darray search_radius(
      py::array const pts, py::float_ const radius, bool const sort) const {
    return impl_->search_radius(pts, radius, sort);
  }

  inline void search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      darray& nns,
      bool const sort) const {
    impl_->search_radius(pts, radius, e, nns, sort);
  }

  inline darray search_radius(
      py::array const pts,
      py::float_ const radius,
      py::float_ const e,
      bool const sort) const {
    return impl_->search_radius(pts, radius, e, sort);
  }

  inline void search_box(py::array const boxes, darray& indices) const {
    impl_->search_box(boxes, indices);
  }

  inline darray search_box(py::array const boxes) const {
    return impl_->search_box(boxes);
  }

  inline py::float_ metric(py::float_ const v) const {
    return impl_->metric(v);
  }

  inline void save(std::iostream& stream) const { impl_->save(stream); }

  inline size_type sdim() const { return impl_->sdim(); }

  inline size_type npts() const { return impl_->npts(); }

  inline py::buffer_info request() const { return impl_->request(); }

  inline std::string metric_string() const { return impl_->metric_string(); }

  inline std::string repr() const { return impl_->repr(); }

  inline py::dtype dtype_index() const { return impl_->dtype_index(); }

  inline py::dtype dtype_scalar() const { return impl_->dtype_scalar(); }

  inline py::dtype dtype_neighbor() const { return impl_->dtype_neighbor(); }

 private:
  std::unique_ptr<internal::kd_tree_base> impl_;
};

class pkd_header {
  // A small signature inspired by the PNG format.
  static constexpr std::array<char, 4> signature{
      static_cast<char>(0x89), 'P', 'K', 'D'};

  // A version to check if the file is compatible with the current
  // implementation.
  static constexpr std::uint32_t version{1};

 public:
  static metric_t read(std::iostream& stream) {
    pico_tree::internal::stream_wrapper wrapper(stream);

    std::array<char, 4> s;
    wrapper.read(s);

    if (s != signature) {
      throw std::runtime_error("unexpected header signature");
    }

    std::uint32_t v;
    wrapper.read(v);

    if (v != version) {
      throw std::runtime_error("unsupported header version");
    }

    std::string metric_string;
    wrapper.read(metric_string);
    return metric_from_string(metric_string);
  }

  static void write(std::string const& metric_string, std::iostream& stream) {
    pico_tree::internal::stream_wrapper wrapper(stream);

    wrapper.write(signature);
    wrapper.write(version);
    wrapper.write(metric_string);
  }

 private:
  static metric_t metric_from_string(std::string const& s) {
    if (s == string_traits<pico_tree::metric_l1>::type_string()) {
      return metric_t::l1;
    } else if (
        s == string_traits<pico_tree::metric_l2_squared>::type_string()) {
      return metric_t::l2_squared;
    } else if (s == string_traits<pico_tree::metric_linf>::type_string()) {
      return metric_t::linf;
    }

    throw std::runtime_error("unexpected metric string");
  }
};

inline kd_tree load_kd_tree(py::array space, std::string const& filename) {
  std::fstream stream = pico_tree::internal::open_stream(
      filename, std::ios::in | std::ios::binary);
  return kd_tree(space, pkd_header::read(stream), stream);
}

inline void save_kd_tree(kd_tree const& tree, std::string const& filename) {
  std::fstream stream = pico_tree::internal::open_stream(
      filename, std::ios::out | std::ios::binary);
  pico_tree::internal::stream_wrapper wrapper(stream);
  pkd_header::write(tree.metric_string(), stream);
  tree.save(stream);
}

}  // namespace pyco_tree
