#pragma once

#include "pico_tree/internal/kd_tree_builder.hpp"
#include "pico_understory/internal/rkd_tree_hh_data.hpp"

namespace pico_tree::internal {

template <typename RKdTreeHhData_, Size Dim_>
class BuildRKdTree {
  using NodeType = typename RKdTreeHhData_::NodeType;
  using KdTreeDataType = KdTreeData<NodeType, Dim_>;

 public:
  using RKdTreeDataType = RKdTreeHhData_;

  template <
      typename SpaceWrapper_,
      typename Stop_,
      typename Bounds_,
      typename Rule_>
  std::vector<RKdTreeDataType> operator()(
      SpaceWrapper_ space,
      splitter_stop_condition_t<Stop_> const& stop_condition,
      splitter_start_bounds_t<Bounds_> const& start_bounds,
      splitter_rule_t<Rule_> const& rule,
      Size forest_size) {
    assert(forest_size > 0);

    using SpaceWrapperType = typename RKdTreeDataType::SpaceWrapperType;
    using BuildKdTreeType = BuildKdTree<KdTreeDataType, Dim_>;

    std::vector<RKdTreeDataType> trees;
    trees.reserve(forest_size);
    for (std::size_t i = 0; i < forest_size; ++i) {
      auto r = RKdTreeDataType::RandomRotation(space);
      auto s = RKdTreeDataType::RotateSpace(r, space);
      auto t = BuildKdTreeType()(
          SpaceWrapperType(s), stop_condition, start_bounds, rule);
      trees.push_back({std::move(r), std::move(s), std::move(t)});
    }
    return trees;
  }
};

}  // namespace pico_tree::internal
