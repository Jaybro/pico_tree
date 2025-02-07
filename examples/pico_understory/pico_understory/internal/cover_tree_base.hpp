#pragma once

namespace pico_tree::internal {

//! \brief This class contains (what we will call) the "leveling base" of the
//! tree.
//! \details It determines how fast the levels of the tree increase or
//! decrease. When we raise "base" to the power of a certain natural number,
//! that exponent represents the active level of the tree.
//!
//! The papers are written using a base of 2, but for performance reasons they
//! use a base of 1.3.
template <typename Scalar_>
struct base {
  template <typename Node_>
  inline Scalar_ cover_distance(Node_ const& n) const {
    return std::pow(value, n.level);
  }

  //! Child distance is also the seperation distance.
  template <typename Node_>
  inline Scalar_ child_distance(Node_ const& n) const {
    return std::pow(value, n.level - Scalar_(1.0));
  }

  template <typename Node_>
  inline Scalar_ parent_distance(Node_ const& n) const {
    return std::pow(value, n.level + Scalar_(1.0));
  }

  inline Scalar_ level(Scalar_ const dst) const {
    return std::log(dst) / std::log(value);
  }

  Scalar_ value;
};

}  // namespace pico_tree::internal
