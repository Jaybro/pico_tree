#pragma once

#include <type_traits>

namespace pico_tree {

//! \brief SpaceTraits provides an interface for the different space types that
//! are supported by PicoTree.
//! \tparam Space_ Any of the space types supported by SpaceTraits.
template <typename Space_>
struct SpaceTraits;

//! \brief Provides an interface for std::reference_wrapper<Space_>.
//! \details If Space_ is already a reference type, such as with an Eigen::Map<>
//! or cv::Mat, then using this specialization won't make much sense.
//! \tparam Space_ Any of the space types supported by SpaceTraits.
template <typename Space_>
struct SpaceTraits<std::reference_wrapper<Space_>>
    : public SpaceTraits<std::remove_const_t<Space_>> {
  //! \brief The SpaceType of these traits.
  //! \details This overrides the SpaceType of the base class. No other
  //! interface changes are required as an std::reference_wrapper can implicitly
  //! be converted to its contained reference, which is a reference to an object
  //! of the exact same type as that of the SpaceType of the base class.
  using SpaceType = std::reference_wrapper<Space_>;
};

}  // namespace pico_tree
