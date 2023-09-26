#pragma once

namespace pico_tree {

//! \brief PointTraits provides an interface for the different point types that
//! are supported by PicoTree.
//! \details Examples of how a PointTraits can be created and used are linked
//! below.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \see PointTraits<Scalar_[Dim_]>
//! \see SpaceTraits<std::vector<Point_, Allocator_>>
template <typename Point_>
struct PointTraits;

}  // namespace pico_tree
