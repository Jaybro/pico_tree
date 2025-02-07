#pragma once

#include <pico_tree/core.hpp>
#include <pico_tree/space_traits.hpp>
#include <utility>

namespace pico_tree {

namespace internal {

template <typename Space_>
class dynamic_space_base {
 public:
  using size_type = pico_tree::size_t;

  inline explicit dynamic_space_base(Space_ space)
      : space_(std::move(space)),
        sdim_(pico_tree::space_traits<Space_>::sdim(space_)) {}

  inline size_type sdim() const { return sdim_; }

 protected:
  Space_ space_;
  size_type sdim_;
};

}  // namespace internal

//!
template <typename Space_>
class dynamic_space : protected internal::dynamic_space_base<Space_> {
 public:
  using internal::dynamic_space_base<Space_>::dynamic_space_base;
  using internal::dynamic_space_base<Space_>::sdim;
  using internal::dynamic_space_base<Space_>::space_;

  inline operator Space_ const&() const { return space_; }
  inline operator Space_&() { return space_; }
};

template <typename Space_>
class dynamic_space<std::reference_wrapper<Space_>>
    : protected internal::dynamic_space_base<std::reference_wrapper<Space_>> {
 public:
  using internal::dynamic_space_base<
      std::reference_wrapper<Space_>>::dynamic_space_base;
  using internal::dynamic_space_base<std::reference_wrapper<Space_>>::sdim;
  using internal::dynamic_space_base<std::reference_wrapper<Space_>>::space_;

  inline operator Space_ const&() const { return space_; }
  inline operator Space_&() { return space_; }
};

template <typename Space_>
struct space_traits<dynamic_space<Space_>> : public space_traits<Space_> {
  using space_type = dynamic_space<Space_>;
  using size_type = pico_tree::size_t;
  static size_type constexpr dim = pico_tree::dynamic_size;

  inline static size_type sdim(space_type const& space) { return space.sdim(); }
};

}  // namespace pico_tree
