#pragma once

#include "sequence.hpp"

namespace pico_tree {

namespace internal {

template <typename Box>
struct BoxTraits;

template <typename Derived>
class BoxBase {
 private:
 public:
  using ScalarType = typename BoxTraits<Derived>::ScalarType;
  static constexpr int Dim = BoxTraits<Derived>::Dim;

  //! \brief Checks if \p x is contained. A point on the edge considered inside
  //! the box.
  inline bool Contains(ScalarType const* const x) const {
    // We use derived().size() which includes the constexpr part. Otherwise a
    // small trait needs to be written.
    for (int i = 0; i < derived().size(); ++i) {
      if (min()[i] > x[i] || max()[i] < x[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename OtherDerived>
  inline bool Contains(BoxBase<OtherDerived> const& x) const {
    return Contains(x.derived().min().container().data()) &&
           Contains(x.derived().max().container().data());
  }

  //! Returns a const reference to the derived class.
  inline Derived const& derived() const {
    return *static_cast<Derived const*>(this);
  }
  //! Returns a reference to the derived class.
  inline Derived& derived() { return *static_cast<Derived*>(this); }

  inline ScalarType const* const min() const { return derived().min(); }
  inline ScalarType* min() { return derived().min(); }
  inline ScalarType const* const max() const { return derived().max(); }
  inline ScalarType* max() { return derived().max(); }
  inline int size() const { return derived().size(); }

 private:
};

//! \brief A SequenceBox can be used as a bounding box. It uses a Sequence for
//! storing the min and max coordinate of the box.
template <typename Scalar_, int Dim_>
struct SequenceBox : public BoxBase<SequenceBox<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline Sequence<Scalar_, Dim_> const& min() const { return min_; }
  inline Sequence<Scalar_, Dim_>& min() { return min_; }
  inline Sequence<Scalar_, Dim_> const& max() const { return max_; }
  inline Sequence<Scalar_, Dim_>& max() { return max_; }
  inline int constexpr size() const { return min_.size(); }

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

  inline BoxMap(ScalarType const* const min, ScalarType const* const max, int)
      : min_(min), max_(max) {}

  inline ScalarType const* const min() const { return min_; }
  inline ScalarType* min() { return min_; }
  inline ScalarType const* const max() const { return max_; }
  inline ScalarType* max() { return max_; }
  inline static int constexpr size() { return Dim_; }

 protected:
  ScalarType const* const min_;
  ScalarType const* const max_;
};

template <typename Scalar_>
class BoxMap<Scalar_, kDynamicDim>
    : public BoxBase<BoxMap<Scalar_, kDynamicDim>> {
 public:
  using ScalarType = Scalar_;
  static int constexpr Dim = kDynamicDim;

  inline BoxMap(
      ScalarType const* const min, ScalarType const* const max, int size)
      : min_(min), max_(max), size_(size) {}

  inline ScalarType const* const min() const { return min_; }
  inline ScalarType* min() { return min_; }
  inline ScalarType const* const max() const { return max_; }
  inline ScalarType* max() { return max_; }
  inline int size() const { return size_; }

 protected:
  ScalarType const* const min_;
  ScalarType const* const max_;
  int size_;
};

template <typename Scalar_, int Dim_>
struct BoxTraits<SequenceBox<Scalar_, Dim_>> {
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
