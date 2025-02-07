#pragma once

#include "pico_tree/internal/kd_tree_builder.hpp"
#include "pico_understory/internal/rkd_tree_hh_data.hpp"

namespace pico_tree::internal {

template <typename RKdTreeHhData_, size_t Dim_>
class build_rkd_tree {
  using node_type = typename RKdTreeHhData_::node_type;
  using kd_tree_data_type = kd_tree_data<node_type, Dim_>;

 public:
  using rkd_tree_data_type = RKdTreeHhData_;

  template <
      typename SpaceWrapper_,
      typename Stop_,
      typename Bounds_,
      typename Rule_>
  std::vector<rkd_tree_data_type> operator()(
      SpaceWrapper_ space,
      splitter_stop_condition_t<Stop_> const& stop_condition,
      splitter_start_bounds_t<Bounds_> const& start_bounds,
      splitter_rule_t<Rule_> const& rule,
      size_t forest_size) {
    assert(forest_size > 0);

    using space_wrapper_type = typename rkd_tree_data_type::space_wrapper_type;
    using build_kd_tree_type = build_kd_tree<kd_tree_data_type, Dim_>;

    std::vector<rkd_tree_data_type> trees;
    trees.reserve(forest_size);
    for (std::size_t i = 0; i < forest_size; ++i) {
      auto r = rkd_tree_data_type::random_rotation(space);
      auto s = rkd_tree_data_type::rotate_space(r, space);
      auto t = build_kd_tree_type()(
          space_wrapper_type(s), stop_condition, start_bounds, rule);
      trees.push_back({std::move(r), std::move(s), std::move(t)});
    }
    return trees;
  }
};

}  // namespace pico_tree::internal
