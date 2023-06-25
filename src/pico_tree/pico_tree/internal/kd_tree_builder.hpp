#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

enum class SplittingRule {
  //! \brief Splits a node on the median point along the dimension of the node's
  //! box longest side. This rule is also known as the standard split rule.
  //! \details This rule builds a tree in O(n log n) time on average. It's
  //! generally slower compared to kSlidingMidpoint but results in a
  //! balanced KdTree.
  kLongestMedian,
  //! \brief Splits a node's box halfway the dimension of its longest side. The
  //! first dimension is chosen if multiple sides share being the longest. This
  //! splitting rule can result in empty nodes.
  //! \details The aspect ratio of the split is at most 2:1.
  //! \see kSlidingMidpoint
  kMidpoint,
  //! \brief Splits a node's box halfway the dimension of its longest side. The
  //! first dimension is chosen if multiple sides share being the longest. In
  //! case the split results in an empty sub-node, the split is adjusted to
  //! include a single point into that sub-node.
  //! \details Based on the paper "It's okay to be skinny, if your friends are
  //! fat". The aspect ratio of the split is at most 2:1 unless that results in
  //! an empty sub-node.
  //!
  //! * http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf
  //!
  //! This splitter can be used to answer an approximate nearest neighbor query
  //! in O(1/e^d log n) time.
  //!
  //! The tree is build in O(n log n) time and results in a tree that is both
  //! faster to build and generally faster to query as compared to
  //! kLongestMedian.
  kSlidingMidpoint
};

namespace internal {

//! \see SplittingRule::kLongestMedian
template <typename Space_>
class SplitterLongestMedian {
  using ScalarType = typename Space_::ScalarType;
  using SizeType = Size;
  using SpaceType = Space_;
  using BoxType = Box<ScalarType, Space_::Dim>;

 public:
  SplitterLongestMedian(SpaceType const& space) : space_{space} {}

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
  SpaceType const& space_;
};

//! \see SplittingRule::kMidpoint
template <typename Space_>
class SplitterMidpoint {
  using ScalarType = typename Space_::ScalarType;
  using SizeType = Size;
  using SpaceType = Space_;
  using BoxType = Box<ScalarType, Space_::Dim>;

 public:
  SplitterMidpoint(SpaceType const& space) : space_{space} {}

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
  }

 private:
  SpaceType const& space_;
};

//! \see SplittingRule::kSlidingMidpoint
template <typename Space_>
class SplitterSlidingMidpoint {
  using ScalarType = typename Space_::ScalarType;
  using SizeType = Size;
  using SpaceType = Space_;
  using BoxType = Box<ScalarType, Space_::Dim>;

 public:
  SplitterSlidingMidpoint(SpaceType const& space) : space_{space} {}

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
  SpaceType const& space_;
};

template <SplittingRule Rule_>
struct SplittingRuleTraits;

template <>
struct SplittingRuleTraits<SplittingRule::kLongestMedian> {
  template <typename Space_>
  using SplitterType = SplitterLongestMedian<Space_>;
};

template <>
struct SplittingRuleTraits<SplittingRule::kMidpoint> {
  template <typename Space_>
  using SplitterType = SplitterMidpoint<Space_>;
};

template <>
struct SplittingRuleTraits<SplittingRule::kSlidingMidpoint> {
  template <typename Space_>
  using SplitterType = SplitterSlidingMidpoint<Space_>;
};

//! \brief This class provides the build algorithm of the KdTree. How the
//! KdTree will be build depends on the Splitter template argument.
template <typename Space_, SplittingRule SplittingRule_, typename KdTreeData_>
class BuildKdTreeImpl {
 public:
  using IndexType = typename KdTreeData_::IndexType;
  using ScalarType = typename KdTreeData_::ScalarType;
  using SizeType = Size;
  using SpaceType = Space_;
  using BoxType = Box<ScalarType, KdTreeData_::Dim>;
  using SplitterType = typename SplittingRuleTraits<
      SplittingRule_>::template SplitterType<Space_>;
  using KdTreeDataType = KdTreeData_;
  using NodeType = typename KdTreeDataType::NodeType;
  using NodeAllocatorType = typename KdTreeDataType::NodeAllocatorType;

  BuildKdTreeImpl(
      SpaceType const& space,
      SizeType const max_leaf_size,
      std::vector<IndexType>& indices,
      NodeAllocatorType& allocator)
      : space_(space),
        max_leaf_size_(
            static_cast<typename std::vector<IndexType>::difference_type>(
                max_leaf_size)),
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
    if ((end - begin) <= max_leaf_size_) {
      node->data.leaf.begin_idx =
          static_cast<IndexType>(begin - indices_.begin());
      node->data.leaf.end_idx = static_cast<IndexType>(end - indices_.begin());
      node->left = nullptr;
      node->right = nullptr;
      // Keep the original box in case it was empty.
      if (end > begin) {
        ComputeBoundingBox(begin, end, box);
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

      SetBranch(box, right, split_dim, *node);

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

  inline void SetBranch(
      BoxType const& left,
      BoxType const& right,
      SizeType const split_dim,
      KdTreeNodeEuclidean<IndexType, ScalarType>& node) const {
    node.data.branch.split_dim = static_cast<int>(split_dim);
    node.data.branch.left_max = left.max(split_dim);
    node.data.branch.right_min = right.min(split_dim);
  }

  inline void SetBranch(
      BoxType const& left,
      BoxType const& right,
      SizeType const split_dim,
      KdTreeNodeTopological<IndexType, ScalarType>& node) const {
    node.data.branch.split_dim = static_cast<int>(split_dim);
    node.data.branch.left_min = left.min(split_dim);
    node.data.branch.left_max = left.max(split_dim);
    node.data.branch.right_min = right.min(split_dim);
    node.data.branch.right_max = right.max(split_dim);
  }

  SpaceType const& space_;
  typename std::vector<IndexType>::difference_type const max_leaf_size_;
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

template <
    typename Space_,
    typename Metric_,
    SplittingRule SplittingRule_,
    typename Index_>
class BuildKdTree {
  using IndexType = Index_;
  using ScalarType = typename Space_::ScalarType;
  static Size constexpr Dim = Space_::Dim;
  //! \brief Node type based on Metric_::SpaceTag.
  using NodeType = typename KdTreeSpaceTagTraits<
      typename Metric_::SpaceTag>::template NodeType<IndexType, ScalarType>;

 public:
  using KdTreeDataType = KdTreeData<NodeType, Dim>;

  //! \brief Construct a KdTree given \p points , \p max_leaf_size and
  //! SplitterType.
  KdTreeDataType operator()(Space_ const& space, Size const max_leaf_size) {
    assert(space.size() > 0);
    assert(max_leaf_size > 0);

    using BuildKdTreeImplType =
        BuildKdTreeImpl<Space_, SplittingRule_, KdTreeDataType>;
    using NodeAllocatorType = typename KdTreeDataType::NodeAllocatorType;
    using BoxType = Box<ScalarType, Dim>;

    std::vector<IndexType> indices(space.size());
    std::iota(indices.begin(), indices.end(), 0);
    BoxType root_box = space.ComputeBoundingBox();
    NodeAllocatorType allocator;
    NodeType* root_node =
        BuildKdTreeImplType{space, max_leaf_size, indices, allocator}(root_box);

    return KdTreeDataType{
        std::move(indices), root_box, std::move(allocator), root_node};
  }
};

}  // namespace internal

}  // namespace pico_tree