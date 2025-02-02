#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point_wrapper.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

template <typename Derived_>
struct splitter_rule_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_rule_t() = default;
  constexpr explicit splitter_rule_t(splitter_rule_t const&) = default;
  constexpr explicit splitter_rule_t(splitter_rule_t&&) = default;
};

//! \brief Splits a node on the median point along the dimension of the node's
//! box longest side. This rule is also known as the standard split rule.
//! \details This rule builds a tree in O(n log n) time on average. It's
//! generally slower compared to sliding_midpoint_max_side_t but results in a
//! balanced KdTree.
struct median_max_side_t : public splitter_rule_t<median_max_side_t> {
  constexpr explicit median_max_side_t() = default;
};

//! \brief Splits a node's box halfway the dimension of its longest side. The
//! first dimension is chosen if multiple sides share being the longest. This
//! splitting rule can result in empty nodes.
//! \details The aspect ratio of the split is at most 2:1.
struct midpoint_max_side_t : public splitter_rule_t<midpoint_max_side_t> {
  constexpr explicit midpoint_max_side_t() = default;
};

//! \brief Splits a node's box halfway the dimension of its longest side. The
//! first dimension is chosen if multiple sides share being the longest. In case
//! the split results in an empty sub-node, the split is adjusted to include a
//! single point into that sub-node.
//! \details Based on the paper "It's okay to be skinny, if your friends are
//! fat". The aspect ratio of the split is at most 2:1 unless that results in an
//! empty sub-node.
//!
//! * http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf
//!
//! This splitter can be used to answer an approximate nearest neighbor query in
//! O(1/e^d log n) time.
//!
//! The tree is build in O(n log n) time and results in a tree that is both
//! faster to build and generally faster to query as compared to
//! median_max_side_t.
struct sliding_midpoint_max_side_t
    : public splitter_rule_t<sliding_midpoint_max_side_t> {
  constexpr explicit sliding_midpoint_max_side_t() = default;
};

//! \copydoc median_max_side_t
inline constexpr median_max_side_t median_max_side{};

//! \copydoc midpoint_max_side_t
inline constexpr midpoint_max_side_t midpoint_max_side{};

//! \copydoc sliding_midpoint_max_side_t
inline constexpr sliding_midpoint_max_side_t sliding_midpoint_max_side{};

template <typename Derived_>
struct splitter_stop_condition_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_stop_condition_t() = default;
  constexpr explicit splitter_stop_condition_t(
      splitter_stop_condition_t const&) = default;
  constexpr explicit splitter_stop_condition_t(splitter_stop_condition_t&&) =
      default;
};

//! \brief The maximum number of points allowed in a leaf node.
struct max_leaf_size_t : public splitter_stop_condition_t<max_leaf_size_t> {
  constexpr max_leaf_size_t(std::size_t v) : value(v) { assert(value > 0); }

  std::size_t value;
};

//! \brief The maximum depth allowed for a leaf node. A depth of zero means that
//! the root node becomes a leaf node.
struct max_leaf_depth_t : public splitter_stop_condition_t<max_leaf_depth_t> {
  constexpr max_leaf_depth_t(std::size_t v) : value(v) {}

  std::size_t value;
};

template <typename Derived_>
struct splitter_start_bounds_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_start_bounds_t() = default;
  constexpr explicit splitter_start_bounds_t(splitter_start_bounds_t const&) =
      default;
  constexpr explicit splitter_start_bounds_t(splitter_start_bounds_t&&) =
      default;
};

struct bounds_from_space_t
    : public splitter_start_bounds_t<bounds_from_space_t> {
  constexpr explicit bounds_from_space_t() = default;
};

inline constexpr bounds_from_space_t bounds_from_space{};

template <typename Point_>
struct bounds_t : public splitter_start_bounds_t<bounds_t<Point_>> {
  constexpr explicit bounds_t(Point_ const& min, Point_ const& max)
      : min_(min), max_(max) {}

  constexpr Point_ const& min() const { return min_; }
  constexpr Point_ const& max() const { return max_; }

 private:
  Point_ min_;
  Point_ max_;
};

namespace internal {

//! \copydoc SplittingRule::kLongestMedian
template <typename SpaceWrapper_>
class SplitterLongestMedian {
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using SizeType = Size;
  using BoxType = Box<ScalarType, SpaceWrapper_::Dim>;

 public:
  SplitterLongestMedian(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType const& box,
      RandomAccessIterator_& split,
      SizeType& split_dim,
      ScalarType& split_val) const {
    ScalarType max_delta;
    box.LongestAxis(split_dim, max_delta);

    split = begin + (end - begin) / 2;

    std::nth_element(
        begin,
        split,
        end,
        [this, &split_dim](auto const index_a, auto const index_b) -> bool {
          return space_[index_a][split_dim] < space_[index_b][split_dim];
        });

    split_val = space_[*split][split_dim];
  }

 private:
  SpaceWrapper_ space_;
};

//! \copydoc SplittingRule::kMidpoint
template <typename SpaceWrapper_>
class SplitterMidpoint {
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using SizeType = Size;
  using BoxType = Box<ScalarType, SpaceWrapper_::Dim>;

 public:
  SplitterMidpoint(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType const& box,
      RandomAccessIterator_& split,
      SizeType& split_dim,
      ScalarType& split_val) const {
    ScalarType max_delta;
    box.LongestAxis(split_dim, max_delta);
    split_val = max_delta * ScalarType(0.5) + box.min(split_dim);

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](auto const index) -> bool {
      return space_[index][split_dim] < split_val;
    };

    split = std::partition(begin, end, comp);
  }

 private:
  SpaceWrapper_ space_;
};

//! \copydoc SplittingRule::kSlidingMidpoint
template <typename SpaceWrapper_>
class SplitterSlidingMidpoint {
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using SizeType = Size;
  using BoxType = Box<ScalarType, SpaceWrapper_::Dim>;

 public:
  SplitterSlidingMidpoint(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType const& box,
      RandomAccessIterator_& split,
      SizeType& split_dim,
      ScalarType& split_val) const {
    ScalarType max_delta;
    box.LongestAxis(split_dim, max_delta);
    split_val = max_delta / ScalarType(2.0) + box.min(split_dim);

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](auto const index) -> bool {
      return space_[index][split_dim] < split_val;
    };

    split = std::partition(begin, end, comp);

    // If it happens that either all points are on the left side or right
    // side, one point slides to the other side and we split on the first
    // right value instead of the middle split. In these two cases the split
    // value is unknown and a partial sort is required to obtain it, but also
    // to rearrange all other indices such that they are on their
    // corresponding left or right side.
    if (split == end) {
      split--;
      std::nth_element(
          begin,
          split,
          end,
          [this, &split_dim](auto const index_a, auto const index_b) -> bool {
            return space_[index_a][split_dim] < space_[index_b][split_dim];
          });
      split_val = space_[*split][split_dim];
    } else if (split == begin) {
      split++;
      std::nth_element(
          begin,
          split,
          end,
          [this, &split_dim](auto const index_a, auto const index_b) -> bool {
            return space_[index_a][split_dim] < space_[index_b][split_dim];
          });
      split_val = space_[*split][split_dim];
    }
  }

 private:
  SpaceWrapper_ space_;
};

template <typename Rule_>
struct SplittingRuleTraits;

template <>
struct SplittingRuleTraits<median_max_side_t> {
  template <typename SpaceWrapper_>
  using SplitterType = SplitterLongestMedian<SpaceWrapper_>;
};

template <>
struct SplittingRuleTraits<midpoint_max_side_t> {
  template <typename SpaceWrapper_>
  using SplitterType = SplitterMidpoint<SpaceWrapper_>;
};

template <>
struct SplittingRuleTraits<sliding_midpoint_max_side_t> {
  template <typename SpaceWrapper_>
  using SplitterType = SplitterSlidingMidpoint<SpaceWrapper_>;
};

//! \brief This class provides the build algorithm of the KdTree. How the
//! KdTree will be build depends on the Splitter template argument.
template <
    typename SpaceWrapper_,
    typename Stop_,
    typename Rule_,
    typename KdTreeData_>
class BuildKdTreeImpl {
 public:
  using IndexType = typename KdTreeData_::IndexType;
  using ScalarType = typename KdTreeData_::ScalarType;
  using SizeType = Size;
  using SpaceType = SpaceWrapper_;
  using BoxType = Box<ScalarType, KdTreeData_::Dim>;
  using SplitterType =
      typename SplittingRuleTraits<Rule_>::template SplitterType<SpaceWrapper_>;
  using KdTreeDataType = KdTreeData_;
  using NodeType = typename KdTreeDataType::NodeType;
  using NodeAllocatorType = typename KdTreeDataType::NodeAllocatorType;

  BuildKdTreeImpl(
      SpaceType const& space,
      SizeType const stop_value,
      std::vector<IndexType>& indices,
      NodeAllocatorType& allocator)
      : space_(space),
        stop_value_(static_cast<IndexType>(stop_value)),
        splitter_(space_),
        indices_(indices),
        allocator_(allocator) {}

  //! \brief Creates the full set of nodes for a KdTree.
  inline NodeType* operator()(BoxType const& root_box) {
    BoxType box(root_box);
    return SplitIndices(0, indices_.begin(), indices_.end(), box);
  }

 private:
  //! \brief Creates a tree node for a range of indices, splits the range in
  //! two and recursively does the same for each sub set of indices until the
  //! index range size is less than or equal to max_leaf_size_.
  //! \details While descending the tree we split nodes based on the root box
  //! until leaf nodes are reached. Inside the leaf nodes the boxes are updated
  //! to be the bounding boxes of the points they contain. While unwinding the
  //! recursion we update the split information for each branch node based on
  //! merging leaf nodes. Since the updated split informaton based on the leaf
  //! nodes can have smaller bounding boxes than the original ones, we can
  //! improve query times.
  template <typename RandomAccessIterator_>
  inline NodeType* SplitIndices(
      IndexType const depth,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType& box) const {
    NodeType* node = allocator_.Allocate();
    //
    if (is_leaf(depth, begin, end)) {
      node->data.leaf.begin_idx =
          static_cast<IndexType>(begin - indices_.begin());
      node->data.leaf.end_idx = static_cast<IndexType>(end - indices_.begin());
      node->left = nullptr;
      node->right = nullptr;
      // Keep the original box in case it was empty. This can only happen with
      // the midpoint split.
      // TODO Optimize node usage for midpoint split (avoid empty nodes).
      if constexpr (std::is_same_v<Rule_, midpoint_max_side_t>) {
        if (end > begin) {
          ComputeBoundingBox(begin, end, box);
        }
      }
    } else {
      // split equals end for the left branch and begin for the right branch.
      RandomAccessIterator_ split;
      SizeType split_dim;
      ScalarType split_val;
      splitter_(depth, begin, end, box, split, split_dim, split_val);

      BoxType right = box;
      // Argument box will function as the left bounding box until we merge
      // left and right again at the end of this code section.
      box.max(split_dim) = split_val;
      right.min(split_dim) = split_val;

      node->left = SplitIndices(depth + 1, begin, split, box);
      node->right = SplitIndices(depth + 1, split, end, right);

      node->SetBranch(box, right, split_dim);

      // Merges both child boxes. We can expect any of the min max values to
      // change except for the ones of split_dim.
      box.Fit(right);
    }

    return node;
  }

  template <typename RandomAccessIterator_>
  inline void ComputeBoundingBox(
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType& box) const {
    box.FillInverseMax();
    for (; begin < end; ++begin) {
      box.Fit(space_[*begin]);
    }
  }

  template <typename RandomAccessIterator_>
  inline bool is_leaf(
      IndexType depth,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end) const {
    if constexpr (std::is_same_v<Stop_, max_leaf_size_t>) {
      return (end - begin) <= stop_value_;
    } else {
      return depth == stop_value_;
    }
  }

  SpaceType const& space_;
  IndexType const stop_value_;
  SplitterType splitter_;
  std::vector<IndexType>& indices_;
  NodeAllocatorType& allocator_;
};

//! \brief KdTree meta information depending on the SpaceTag_ template argument.
template <typename SpaceTag_>
struct KdTreeSpaceTagTraits;

//! \brief KdTree meta information for the EuclideanSpaceTag.
template <>
struct KdTreeSpaceTagTraits<EuclideanSpaceTag> {
  //! \brief Supported node type.
  template <typename Index_, typename Scalar_>
  using NodeType = KdTreeNodeEuclidean<Index_, Scalar_>;
};

//! \brief KdTree meta information for the TopologicalSpaceTag.
template <>
struct KdTreeSpaceTagTraits<TopologicalSpaceTag> {
  //! \brief Supported node type.
  template <typename Index_, typename Scalar_>
  using NodeType = KdTreeNodeTopological<Index_, Scalar_>;
};

template <typename KdTreeData_, Size Dim_>
class BuildKdTree {
  using IndexType = typename KdTreeData_::IndexType;
  using ScalarType = typename KdTreeData_::ScalarType;
  using NodeType = typename KdTreeData_::NodeType;
  using BoxType = Box<ScalarType, Dim_>;

 public:
  using KdTreeDataType = KdTreeData_;

  //! \brief Construct a KdTree given \p points , \p max_leaf_size and
  //! SplitterType.
  template <
      typename SpaceWrapper_,
      typename Stop_,
      typename Bounds_,
      typename Rule_>
  KdTreeDataType operator()(
      SpaceWrapper_ space,
      splitter_stop_condition_t<Stop_> const& stop_condition,
      splitter_start_bounds_t<Bounds_> const& start_bounds,
      splitter_rule_t<Rule_> const&) {
    static_assert(
        std::is_same_v<ScalarType, typename SpaceWrapper_::ScalarType>);
    static_assert(Dim_ == SpaceWrapper_::Dim);
    assert(space.size() > 0);

    using BuildKdTreeImplType =
        BuildKdTreeImpl<SpaceWrapper_, Stop_, Rule_, KdTreeDataType>;
    using NodeAllocatorType = typename KdTreeDataType::NodeAllocatorType;

    std::vector<IndexType> indices(space.size());
    std::iota(indices.begin(), indices.end(), 0);
    BoxType root_box = box_from_bounds(space, start_bounds.derived());
    NodeAllocatorType allocator;
    NodeType* root_node = BuildKdTreeImplType{
        space, stop_condition.derived().value, indices, allocator}(root_box);

    return KdTreeDataType{
        std::move(indices), root_box, std::move(allocator), root_node};
  }

 private:
  template <typename SpaceWrapper_>
  BoxType box_from_bounds(SpaceWrapper_ space, bounds_from_space_t) const {
    return space.ComputeBoundingBox();
  }

  template <typename SpaceWrapper_, typename Point_>
  BoxType box_from_bounds(
      SpaceWrapper_ space, bounds_t<Point_> const& bounds) const {
    internal::PointWrapper<Point_> min(bounds.min());
    internal::PointWrapper<Point_> max(bounds.max());
    BoxType box(space.sdim());

    box.FillInverseMax();
    box.Fit(min.data());
    box.Fit(max.data());
    return box;
  }
};

}  // namespace internal

}  // namespace pico_tree
