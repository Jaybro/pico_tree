#pragma once

#include <array>

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

//! \brief BoxTraits exposes metadata for each of the different box types.
//! \see Box
//! \see BoxMap
template <typename Box_>
struct BoxTraits;

//! \brief BoxBase exposes various box utilities.
//! \details CRTP based base class for any of the box child classes.
//! \tparam Derived Any of the box child classes.
template <typename Derived>
class BoxBase {
 public:
  using ScalarType = typename BoxTraits<Derived>::ScalarType;
  using SizeType = Size;
  static constexpr SizeType Dim = BoxTraits<Derived>::Dim;
  static_assert(Dim == kDynamicDim || Dim > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  //! \brief Returns true if \p x is contained. A point on the edge considered
  //! inside the box.
  inline bool Contains(ScalarType const* x) const {
    // We use derived().size() which includes the constexpr part. Otherwise a
    // small trait needs to be written.
    for (SizeType i = 0; i < derived().size(); ++i) {
      if (min(i) > x[i] || max(i) < x[i]) {
        return false;
      }
    }
    return true;
  }

  //! \brief Returns true if \p x is contained. When the input box is identical,
  //! it is considered contained.
  template <typename OtherDerived>
  inline bool Contains(BoxBase<OtherDerived> const& x) const {
    return Contains(x.min()) && Contains(x.max());
  }

  //! \brief Sets the values of min and max to be an inverted maximum bounding
  //! box.
  //! \details The values for min and max are set to respectively the maximum
  //! and minimum possible values for integers or floating points. This is
  //! useful for growing a bounding box in combination with the Update function.
  inline void FillInverseMax() {
    for (SizeType i = 0; i < derived().size(); ++i) {
      min(i) = std::numeric_limits<ScalarType>::max();
      max(i) = std::numeric_limits<ScalarType>::lowest();
    }
  }

  //! \brief See which axis of the box is the longest.
  //! \param p_max_index Output parameter for the index of the longest axis.
  //! \param p_max_value Output parameter for the range of the longest axis.
  inline void LongestAxis(
      SizeType* p_max_index, ScalarType* p_max_value) const {
    *p_max_value = std::numeric_limits<ScalarType>::lowest();

    for (SizeType i = 0; i < derived().size(); ++i) {
      ScalarType const delta = max(i) - min(i);
      if (delta > *p_max_value) {
        *p_max_index = i;
        *p_max_value = delta;
      }
    }
  }

  //! \brief Updates the min and/or max vectors of this box so that it can fit
  //! input point \p x.
  inline void Fit(ScalarType const* x) {
    for (SizeType i = 0; i < derived().size(); ++i) {
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
  template <typename OtherDerived>
  inline void Fit(BoxBase<OtherDerived> const& x) {
    for (SizeType i = 0; i < derived().size(); ++i) {
      if (x.min(i) < min(i)) {
        min(i) = x.min(i);
      }

      if (x.max(i) > max(i)) {
        max(i) = x.max(i);
      }
    }
  }

  //! \brief Returns a const reference to the derived class.
  inline Derived const& derived() const {
    return *static_cast<Derived const*>(this);
  }
  //! \brief Returns a reference to the derived class.
  inline Derived& derived() { return *static_cast<Derived*>(this); }

  inline ScalarType const* min() const noexcept { return derived().min(); }
  inline ScalarType* min() noexcept { return derived().min(); }
  inline ScalarType const* max() const noexcept { return derived().max(); }
  inline ScalarType* max() noexcept { return derived().max(); }
  inline ScalarType const& min(SizeType i) const noexcept {
    return derived().min(i);
  }
  inline ScalarType& min(SizeType i) noexcept { return derived().min(i); }
  inline ScalarType const& max(SizeType i) const noexcept {
    return derived().max(i);
  }
  inline ScalarType& max(SizeType i) noexcept { return derived().max(i); }
  inline SizeType size() const noexcept { return derived().size(); }

 protected:
  //! \private
  BoxBase() = default;
  //! \private
  ~BoxBase() = default;
};

//! \brief An axis aligned box represented by a min and max coordinate.
//! \details This specialization supports a compile time known spatial
//! dimension.
template <typename Scalar_, Size Dim_>
class Box : public BoxBase<Box<Scalar_, Dim_>> {
 public:
  using ScalarType = Scalar_;
  using typename BoxBase<Box<Scalar_, Dim_>>::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline explicit Box(SizeType) {}

  inline ScalarType const* min() const noexcept { return min_.data(); }
  inline ScalarType* min() noexcept { return min_.data(); }
  inline ScalarType const* max() const noexcept { return max_.data(); }
  inline ScalarType* max() noexcept { return max_.data(); }
  inline ScalarType const& min(SizeType i) const { return min_[i]; }
  inline ScalarType& min(SizeType i) { return min_[i]; }
  inline ScalarType const& max(SizeType i) const { return max_[i]; }
  inline ScalarType& max(SizeType i) { return max_[i]; }
  inline SizeType constexpr size() const noexcept { return min_.size(); }

 private:
  //! \brief Minimum box coordinate.
  std::array<ScalarType, Dim> min_;
  //! \brief Maximum box coordinate.
  std::array<ScalarType, Dim> max_;
};

//! \brief An axis aligned box represented by a min and max coordinate.
//! \details This specialization supports a run time known spatial dimension.
template <typename Scalar_>
class Box<Scalar_, kDynamicDim> : public BoxBase<Box<Scalar_, kDynamicDim>> {
 public:
  using ScalarType = Scalar_;
  using typename BoxBase<Box<Scalar_, kDynamicDim>>::SizeType;
  static SizeType constexpr Dim = kDynamicDim;

  inline explicit Box(SizeType size) : min_(size * 2), size_(size) {}

  inline ScalarType const* min() const noexcept { return min_.data(); }
  inline ScalarType* min() noexcept { return min_.data(); }
  inline ScalarType const* max() const noexcept { return min_.data() + size_; }
  inline ScalarType* max() noexcept { return min_.data() + size_; }
  inline ScalarType const& min(SizeType i) const { return min_[i]; }
  inline ScalarType& min(SizeType i) { return min_[i]; }
  inline ScalarType const& max(SizeType i) const { return min_[i + size_]; }
  inline ScalarType& max(SizeType i) { return min_[i + size_]; }
  inline SizeType size() const noexcept { return size_; }

 private:
  //! \brief Minimum and maximum box coordinates contiguous in memory.
  std::vector<ScalarType> min_;
  SizeType size_;
};

//! \brief An axis aligned box represented by a min and max coordinate. It maps
//! raw pointers.
//! \details This specialization supports a compile time known spatial
//! dimension.
template <typename Scalar_, Size Dim_>
class BoxMap : public BoxBase<BoxMap<Scalar_, Dim_>> {
 public:
  using ScalarType = Scalar_;
  using typename BoxBase<BoxMap<Scalar_, Dim_>>::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline BoxMap(ScalarType* min, ScalarType* max, SizeType)
      : min_(min), max_(max) {}

  inline ScalarType const* min() const noexcept { return min_; }
  inline ScalarType* min() noexcept { return min_; }
  inline ScalarType const* max() const noexcept { return max_; }
  inline ScalarType* max() noexcept { return max_; }
  inline ScalarType const& min(SizeType i) const { return min_[i]; }
  inline ScalarType& min(SizeType i) { return min_[i]; }
  inline ScalarType const& max(SizeType i) const { return max_[i]; }
  inline ScalarType& max(SizeType i) { return max_[i]; }
  inline SizeType constexpr size() const noexcept {
    return static_cast<SizeType>(Dim_);
  }

 private:
  ScalarType* min_;
  ScalarType* max_;
};

//! \brief An axis aligned box represented by a min and max coordinate. It maps
//! raw pointers.
//! \details This specialization supports a run time known spatial dimension.
template <typename Scalar_>
class BoxMap<Scalar_, kDynamicDim>
    : public BoxBase<BoxMap<Scalar_, kDynamicDim>> {
 public:
  using ScalarType = Scalar_;
  using typename BoxBase<BoxMap<Scalar_, kDynamicDim>>::SizeType;
  static SizeType constexpr Dim = kDynamicDim;

  inline BoxMap(ScalarType* min, ScalarType* max, SizeType size)
      : min_(min), max_(max), size_(size) {}

  inline ScalarType const* min() const noexcept { return min_; }
  inline ScalarType* min() noexcept { return min_; }
  inline ScalarType const* max() const noexcept { return max_; }
  inline ScalarType* max() noexcept { return max_; }
  inline ScalarType const& min(SizeType i) const { return min_[i]; }
  inline ScalarType& min(SizeType i) { return min_[i]; }
  inline ScalarType const& max(SizeType i) const { return max_[i]; }
  inline ScalarType& max(SizeType i) { return max_[i]; }
  inline SizeType size() const noexcept { return size_; }

 private:
  ScalarType* min_;
  ScalarType* max_;
  SizeType size_;
};

template <typename Scalar_, Size Dim_>
struct BoxTraits<Box<Scalar_, Dim_>> {
  using ScalarType = typename std::remove_const<Scalar_>::type;
  static Size constexpr Dim = Dim_;
};

template <typename Scalar_, Size Dim_>
struct BoxTraits<BoxMap<Scalar_, Dim_>> {
  using ScalarType = typename std::remove_const<Scalar_>::type;
  static Size constexpr Dim = Dim_;
};

}  // namespace internal

}  // namespace pico_tree
