#pragma once

#include <pico_tree/core.hpp>
#include <pico_tree/space_traits.hpp>
#include <utility>

namespace internal {
template <typename Space_>
class DynamicSpaceBase {
 public:
  using SizeType = pico_tree::Size;

  inline explicit DynamicSpaceBase(Space_ space)
      : space_(std::move(space)),
        sdim_(pico_tree::SpaceTraits<Space_>::sdim(space_)) {}

  inline SizeType sdim() const { return sdim_; }

 protected:
  Space_ space_;
  SizeType sdim_;
};

}  // namespace internal

template <typename Space_>
class DynamicSpace : protected internal::DynamicSpaceBase<Space_> {
 public:
  using internal::DynamicSpaceBase<Space_>::DynamicSpaceBase;
  using internal::DynamicSpaceBase<Space_>::sdim;
  using internal::DynamicSpaceBase<Space_>::space_;

  inline operator Space_ const&() const { return space_; }
  inline operator Space_&() { return space_; }
};

template <typename Space_>
class DynamicSpace<std::reference_wrapper<Space_>>
    : protected internal::DynamicSpaceBase<std::reference_wrapper<Space_>> {
 public:
  using internal::DynamicSpaceBase<
      std::reference_wrapper<Space_>>::DynamicSpaceBase;
  using internal::DynamicSpaceBase<std::reference_wrapper<Space_>>::sdim;
  using internal::DynamicSpaceBase<std::reference_wrapper<Space_>>::space_;

  inline operator Space_ const&() const { return space_; }
  inline operator Space_&() { return space_; }
};

namespace pico_tree {

template <typename Space_>
struct SpaceTraits<DynamicSpace<Space_>> : public SpaceTraits<Space_> {
  using SpaceType = DynamicSpace<Space_>;
  using SizeType = pico_tree::Size;
  static SizeType constexpr Dim = pico_tree::kDynamicSize;

  inline static SizeType Sdim(SpaceType const& space) { return space.sdim(); }
};

}  // namespace pico_tree
