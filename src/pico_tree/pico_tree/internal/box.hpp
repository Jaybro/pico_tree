#pragma once

#include "sequence.hpp"

namespace pico_tree {

namespace internal {

template <typename Box_>
struct BoxTraits;

template <typename Derived>
class BoxBase {
 public:
  using ScalarType = typename BoxTraits<Derived>::ScalarType;
  static constexpr int Dim = BoxTraits<Derived>::Dim;

  //! \brief Checks if \p x is contained. A point on the edge considered inside
  //! the box.
  inline bool Contains(ScalarType const* const x) const {
    // We use derived().size() which includes the constexpr part. Otherwise a
    // small trait needs to be written.
    for (std::size_t i = 0; i < derived().size(); ++i) {
      if (min()[i] > x[i] || max()[i] < x[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename OtherDerived>
  inline bool Contains(BoxBase<OtherDerived> const& x) const {
    return Contains(x.min()) && Contains(x.max());
  }

  inline void FillInverseMax() {
    for (std::size_t i = 0; i < derived().size(); ++i) {
      min()[i] = std::numeric_limits<ScalarType>::max();
      max()[i] = std::numeric_limits<ScalarType>::lowest();
    }
  }

  //! \brief See which axis of the box is the longest.
  inline void LongestAxis(int* p_max_index, ScalarType* p_max_value) const {
    *p_max_value = std::numeric_limits<ScalarType>::lowest();

    for (int i = 0; i < static_cast<int>(derived().size()); ++i) {
      ScalarType const delta = max()[i] - min()[i];
      if (delta > *p_max_value) {
        *p_max_index = i;
        *p_max_value = delta;
      }
    }
  }

  inline void Update(ScalarType const* const x) {
    for (std::size_t i = 0; i < derived().size(); ++i) {
      if (x[i] < min()[i]) {
        min()[i] = x[i];
      }
      if (x[i] > max()[i]) {
        max()[i] = x[i];
      }
    }
  }

  template <typename OtherDerived>
  inline void Update(BoxBase<OtherDerived> const& x) {
    for (std::size_t i = 0; i < derived().size(); ++i) {
      if (x.min()[i] < min()[i]) {
        min()[i] = x.min()[i];
      }

      if (x.max()[i] > max()[i]) {
        max()[i] = x.max()[i];
      }
    }
  }

  //! Returns a const reference to the derived class.
  inline Derived const& derived() const {
    return *static_cast<Derived const*>(this);
  }
  //! Returns a reference to the derived class.
  inline Derived& derived() { return *static_cast<Derived*>(this); }

  inline ScalarType const* min() const { return derived().min(); }
  inline ScalarType* min() { return derived().min(); }
  inline ScalarType const* max() const { return derived().max(); }
  inline ScalarType* max() { return derived().max(); }
  inline int size() const { return derived().size(); }
};

//! \brief A SequenceBox can be used as a bounding box. It uses a Sequence for
//! storing the min and max coordinate of the box.
template <typename Scalar_, int Dim_>
class Box : public BoxBase<Box<Scalar_, Dim_>> {
 public:
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline explicit Box(std::size_t size) : min_(size), max_(size) {}

  inline ScalarType const* min() const { return min_.container().data(); }
  inline ScalarType* min() { return min_.container().data(); }
  inline ScalarType const* max() const { return max_.container().data(); }
  inline ScalarType* max() { return max_.container().data(); }
  inline std::size_t constexpr size() const { return min_.size(); }

  // TODO Remove in the future.
  inline Sequence<Scalar_, Dim_> const& min_seq() const { return min_; }
  inline Sequence<Scalar_, Dim_>& min_seq() { return min_; }
  inline Sequence<Scalar_, Dim_> const& max_seq() const { return max_; }
  inline Sequence<Scalar_, Dim_>& max_seq() { return max_; }

 protected:
  //! \brief Minimum box coordinate.
  Sequence<Scalar_, Dim_> min_;
  //! \brief Maximum box coordinate.
  Sequence<Scalar_, Dim_> max_;
};

template <typename Scalar_, int Dim_>
class BoxMap : public BoxBase<BoxMap<Scalar_, Dim_>> {
 public:
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline BoxMap(ScalarType* min, ScalarType* max, std::size_t)
      : min_(min), max_(max) {}

  inline ScalarType const* min() const { return min_; }
  inline ScalarType* min() { return min_; }
  inline ScalarType const* max() const { return max_; }
  inline ScalarType* max() { return max_; }
  inline std::size_t constexpr size() const {
    return static_cast<std::size_t>(Dim_);
  }

 protected:
  ScalarType* min_;
  ScalarType* max_;
};

template <typename Scalar_>
class BoxMap<Scalar_, kDynamicDim>
    : public BoxBase<BoxMap<Scalar_, kDynamicDim>> {
 public:
  using ScalarType = Scalar_;
  static int constexpr Dim = kDynamicDim;

  inline BoxMap(ScalarType* min, ScalarType* max, std::size_t size)
      : min_(min), max_(max), size_(size) {}

  inline ScalarType const* min() const { return min_; }
  inline ScalarType* min() { return min_; }
  inline ScalarType const* max() const { return max_; }
  inline ScalarType* max() { return max_; }
  inline std::size_t size() const { return size_; }

 protected:
  ScalarType* min_;
  ScalarType* max_;
  std::size_t size_;
};

template <typename Scalar_, int Dim_>
struct BoxTraits<Box<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;
};

template <typename Scalar_, int Dim_>
struct BoxTraits<BoxMap<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;
};

}  // namespace internal

}  // namespace pico_tree
