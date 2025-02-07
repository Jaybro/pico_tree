#pragma once

#include <array>
#include <limits>
#include <vector>

#include "pico_tree/core.hpp"

namespace pico_tree::internal {

//! \brief box_traits exposes metadata for each of the different box types.
//! \see box
//! \see box_map
template <typename Box_>
struct box_traits;

//! \brief box_base exposes various box utilities.
//! \details CRTP based base class for any of the box child classes.
//! \tparam Derived_ Any of the box child classes.
template <typename Derived_>
class box_base {
 public:
  using scalar_type = typename box_traits<Derived_>::scalar_type;
  using size_type = size_t;
  static constexpr size_type dim = box_traits<Derived_>::dimension;
  static_assert(dim == dynamic_size || dim > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  //! \brief Returns true if \p x is contained. A point on the edge is
  //! considered inside the box.
  constexpr bool contains(scalar_type const* x) const {
    // We use derived().size() which includes the constexpr part. Otherwise a
    // small trait needs to be written.
    for (size_type i = 0; i < derived().size(); ++i) {
      if (min(i) > x[i] || max(i) < x[i]) {
        return false;
      }
    }
    return true;
  }

  //! \brief Returns true if \p x is contained. When the input box is identical,
  //! it is considered contained.
  template <typename OtherDerived_>
  constexpr bool contains(box_base<OtherDerived_> const& x) const {
    return contains(x.min()) && contains(x.max());
  }

  //! \brief Sets the values of min and max to be an inverted maximum bounding
  //! box.
  //! \details The values for min and max are set to respectively the maximum
  //! and minimum possible values for integers or floating points. This is
  //! useful for growing a bounding box in combination with the Update function.
  constexpr void fill_inverse_max() {
    for (size_type i = 0; i < derived().size(); ++i) {
      min(i) = std::numeric_limits<scalar_type>::max();
      max(i) = std::numeric_limits<scalar_type>::lowest();
    }
  }

  //! \brief See which axis of the box is the longest.
  //! \param p_max_index Output parameter for the index of the longest axis.
  //! \param p_max_value Output parameter for the range of the longest axis.
  constexpr void max_side(
      size_type& p_max_index, scalar_type& p_max_value) const {
    p_max_value = std::numeric_limits<scalar_type>::lowest();

    for (size_type i = 0; i < derived().size(); ++i) {
      scalar_type const delta = max(i) - min(i);
      if (delta > p_max_value) {
        p_max_index = i;
        p_max_value = delta;
      }
    }
  }

  //! \brief Updates the min and/or max vectors of this box so that it can fit
  //! input point \p x.
  constexpr void fit(scalar_type const* x) {
    for (size_type i = 0; i < derived().size(); ++i) {
      if (x[i] < min(i)) {
        min(i) = x[i];
      }
      if (x[i] > max(i)) {
        max(i) = x[i];
      }
    }
  }

  //! \brief Updates the min and/or max vectors of this box so that it can fit
  //! input box \p x.
  template <typename OtherDerived_>
  constexpr void fit(box_base<OtherDerived_> const& x) {
    for (size_type i = 0; i < derived().size(); ++i) {
      if (x.min(i) < min(i)) {
        min(i) = x.min(i);
      }

      if (x.max(i) > max(i)) {
        max(i) = x.max(i);
      }
    }
  }

  //! \brief Returns a const reference to the derived class.
  constexpr Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

  //! \brief Returns a reference to the derived class.
  constexpr Derived_& derived() { return *static_cast<Derived_*>(this); }

  constexpr scalar_type const* min() const noexcept { return derived().min(); }

  constexpr scalar_type* min() noexcept { return derived().min(); }

  constexpr scalar_type const* max() const noexcept { return derived().max(); }

  constexpr scalar_type* max() noexcept { return derived().max(); }

  constexpr scalar_type const& min(size_type i) const noexcept {
    return derived().min(i);
  }

  constexpr scalar_type& min(size_type i) noexcept { return derived().min(i); }

  constexpr scalar_type const& max(size_type i) const noexcept {
    return derived().max(i);
  }

  constexpr scalar_type& max(size_type i) noexcept { return derived().max(i); }

  constexpr size_type size() const noexcept { return derived().size(); }

 protected:
  //! \private
  constexpr box_base() = default;

  //! \private
  constexpr box_base(box_base const&) = default;

  //! \private
  constexpr box_base(box_base&&) = default;

  //! \private
  constexpr box_base& operator=(box_base const&) = default;

  //! \private
  constexpr box_base& operator=(box_base&&) = default;
};

template <typename Scalar_, size_t Dim_>
struct box_storage {
  constexpr explicit box_storage(size_t) {}

  constexpr Scalar_ const* min() const noexcept { return coords_min.data(); }

  constexpr Scalar_* min() noexcept { return coords_min.data(); }

  constexpr Scalar_ const* max() const noexcept { return coords_max.data(); }

  constexpr Scalar_* max() noexcept { return coords_max.data(); }

  std::array<Scalar_, Dim_> coords_min;
  std::array<Scalar_, Dim_> coords_max;
  static size_t constexpr size = Dim_;
};

//! \details This specialization supports a compile time known spatial
//! dimension.
template <typename Scalar_>
struct box_storage<Scalar_, dynamic_size> {
  constexpr explicit box_storage(size_t size) : coords(size * 2), size(size) {}

  constexpr Scalar_ const* min() const noexcept { return coords.data(); }

  constexpr Scalar_* min() noexcept { return coords.data(); }

  constexpr Scalar_ const* max() const noexcept { return coords.data() + size; }

  constexpr Scalar_* max() noexcept { return coords.data() + size; }

  std::vector<Scalar_> coords;
  size_t size;
};

//! \brief An axis aligned box represented by a min and max coordinate.
template <typename Scalar_, size_t Dim_>
class box : public box_base<box<Scalar_, Dim_>> {
 public:
  using scalar_type = Scalar_;
  using typename box_base<box<Scalar_, Dim_>>::size_type;
  static size_type constexpr dimension = Dim_;

  using box_base<box<Scalar_, Dim_>>::box_base;

  constexpr box() : storage_(dimension) {}

  constexpr explicit box(size_type size) : storage_(size) {}

  constexpr scalar_type const* min() const noexcept { return storage_.min(); }

  constexpr scalar_type* min() noexcept { return storage_.min(); }

  constexpr scalar_type const* max() const noexcept { return storage_.max(); }

  constexpr scalar_type* max() noexcept { return storage_.max(); }

  constexpr scalar_type const& min(size_type i) const {
    return storage_.min()[i];
  }

  constexpr scalar_type& min(size_type i) { return storage_.min()[i]; }

  constexpr scalar_type const& max(size_type i) const {
    return storage_.max()[i];
  }

  constexpr scalar_type& max(size_type i) { return storage_.max()[i]; }

  constexpr size_type size() const noexcept { return storage_.size; }

 private:
  box_storage<Scalar_, Dim_> storage_;
};

template <typename Scalar_, size_t Dim_>
struct box_map_storage {
  constexpr box_map_storage(Scalar_* min, Scalar_* max, size_t)
      : min(min), max(max) {}

  Scalar_* min;
  Scalar_* max;
  static size_t constexpr size = Dim_;
};

//! \details This specialization supports a run time known spatial dimension.
template <typename Scalar_>
struct box_map_storage<Scalar_, dynamic_size> {
  constexpr box_map_storage(Scalar_* min, Scalar_* max, size_t size)
      : min(min), max(max), size(size) {}

  Scalar_* min;
  Scalar_* max;
  size_t size;
};

//! \brief An axis aligned box represented by a min and max coordinate. It maps
//! raw pointers.
//! \details This specialization supports a compile time known spatial
//! dimension.
template <typename Scalar_, size_t Dim_>
class box_map : public box_base<box_map<Scalar_, Dim_>> {
 public:
  using scalar_type = std::remove_cv_t<Scalar_>;
  using element_type = Scalar_;
  using typename box_base<box_map<Scalar_, Dim_>>::size_type;
  static size_type constexpr dimension = Dim_;

  constexpr box_map(element_type* min, element_type* max)
      : storage_(min, max, dimension) {}

  constexpr box_map(element_type* min, element_type* max, size_type size)
      : storage_(min, max, size) {}

  constexpr box_map(box_map const&) = delete;

  constexpr box_map& operator=(box_map const&) = delete;

  constexpr element_type* min() const noexcept { return storage_.min; }

  constexpr element_type* max() const noexcept { return storage_.max; }

  constexpr element_type& min(size_type i) const { return storage_.min[i]; }

  constexpr element_type& max(size_type i) const { return storage_.max[i]; }

  constexpr size_type size() const noexcept { return storage_.size; }

 private:
  box_map_storage<Scalar_, Dim_> storage_;
};

template <typename Scalar_, size_t Dim_>
struct box_traits<box<Scalar_, Dim_>> {
  using scalar_type = std::remove_cv_t<Scalar_>;
  static size_t constexpr dimension = Dim_;
};

template <typename Scalar_, size_t Dim_>
struct box_traits<box_map<Scalar_, Dim_>> {
  using scalar_type = std::remove_cv_t<Scalar_>;
  static size_t constexpr dimension = Dim_;
};

}  // namespace pico_tree::internal
