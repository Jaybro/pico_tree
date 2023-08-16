#pragma once

#include "pico_tree/internal/kd_tree_builder.hpp"
#include "rkd_tree_hh_data.hpp"

namespace pico_tree::internal {

template <typename Node_, Size Dim_, SplittingRule SplittingRule_>
class BuildRKdTree {
 public:
  using RKdTreeDataType = RKdTreeHhData<Node_, Dim_>;

  template <typename SpaceWrapper_>
  std::vector<RKdTreeDataType> operator()(
      SpaceWrapper_ space, Size max_leaf_size, Size forest_size) {
    assert(space.size() > 0);
    assert(max_leaf_size > 0);
    assert(forest_size > 0);

    using SpaceWrapperType = typename RKdTreeDataType::SpaceWrapperType;
    using BuildKdTreeType = BuildKdTree<Node_, Dim_, SplittingRule_>;

    std::vector<RKdTreeDataType> trees;
    trees.reserve(forest_size);
    for (std::size_t i = 0; i < forest_size; ++i) {
      auto r = RKdTreeDataType::RandomRotation(space);
      auto s = RKdTreeDataType::RotateSpace(r, space);
      auto t = BuildKdTreeType()(SpaceWrapperType(s), max_leaf_size);
      trees.push_back({std::move(r), std::move(s), std::move(t)});
    }
    return trees;
  }
};

}  // namespace pico_tree::internal
