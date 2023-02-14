#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/stream.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

namespace internal {

//!\brief Binary node base.
template <typename Derived>
struct KdTreeNodeBase {
  //! \brief Returns if the current node is a branch.
  inline bool IsBranch() const { return left != nullptr && right != nullptr; }
  //! \brief Returns if the current node is a leaf.
  inline bool IsLeaf() const { return left == nullptr && right == nullptr; }

  //! \brief Left child.
  Derived* left;
  //! \brief Right child.
  Derived* right;
};

//! \brief Tree leaf.
template <typename Index_>
struct KdTreeLeaf {
  //! \brief Begin of an index range.
  Index_ begin_idx;
  //! \brief End of an index range.
  Index_ end_idx;
};

//! \brief Tree branch.
template <typename Scalar_>
struct KdTreeBranchSplit {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
};

//! \brief Tree branch.
//! \details This branch version allows identifications (wrapping around) by
//! storing the boundaries of the box that corresponds to the current node. The
//! split value allows arbitrary splitting techniques.
template <typename Scalar_>
struct KdTreeBranchRange {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Minimum coordinate value of the left node box for split_dim.
  Scalar_ left_min;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
  //! \brief Maximum coordinate value of the right node box for split_dim.
  Scalar_ right_max;
};

//! \brief NodeData is used to either store branch or leaf information. Which
//! union member is used can be tested with IsBranch() or IsLeaf().
template <typename Leaf, typename Branch>
union KdTreeNodeData {
  //! \brief Union branch data.
  Branch branch;
  //! \brief Union leaf data.
  Leaf leaf;
};

//! \brief KdTree node for a Euclidean space.
template <typename Index_, typename Scalar_>
struct KdTreeNodeEuclidean
    : public KdTreeNodeBase<KdTreeNodeEuclidean<Index_, Scalar_>> {
  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index_>, KdTreeBranchSplit<Scalar_>> data;
};

//! \brief KdTree node for a topological space.
template <typename Index_, typename Scalar_>
struct KdTreeNodeTopological
    : public KdTreeNodeBase<KdTreeNodeTopological<Index_, Scalar_>> {
  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index_>, KdTreeBranchRange<Scalar_>> data;
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

//! \brief The data structure that represents a KdTree.
template <typename Index_, typename Scalar_, Size Dim_, typename Node_>
struct KdTreeData {
  using IndexType = Index_;
  using ScalarType = Scalar_;
  using BoxType = typename internal::Box<ScalarType, Dim_>;
  using NodeType = Node_;
  using NodeAllocatorType = ChunkAllocator<NodeType, 256>;

 private:
  //! \brief Recursively reads the Node and its descendants.
  inline NodeType* ReadNode(internal::Stream* stream) {
    NodeType* node = allocator.Allocate();
    bool is_leaf;
    stream->Read(&is_leaf);

    if (is_leaf) {
      stream->Read(&node->data.leaf);
      node->left = nullptr;
      node->right = nullptr;
    } else {
      stream->Read(&node->data.branch);
      node->left = ReadNode(stream);
      node->right = ReadNode(stream);
    }

    return node;
  }

  //! \brief Recursively writes the Node and its descendants.
  inline void WriteNode(
      NodeType const* const node, internal::Stream* stream) const {
    if (node->IsLeaf()) {
      stream->Write(true);
      stream->Write(node->data.leaf);
    } else {
      stream->Write(false);
      stream->Write(node->data.branch);
      WriteNode(node->left, stream);
      WriteNode(node->right, stream);
    }
  }

  inline void Read(internal::Stream* stream) {
    stream->Read(&indices);
    // The root box gets the correct size from the KdTree constructor.
    stream->Read(root_box.size(), root_box.min());
    stream->Read(root_box.size(), root_box.max());
    root_node = ReadNode(stream);
  }

  inline void Write(internal::Stream* stream) const {
    stream->Write(indices);
    stream->Write(root_box.min(), root_box.size());
    stream->Write(root_box.max(), root_box.size());
    WriteNode(root_node, stream);
  }

 public:
  static KdTreeData Load(internal::Stream* stream) {
    typename BoxType::SizeType sdim;
    stream->Read(&sdim);

    KdTreeData kd_tree_data{{}, BoxType(sdim), NodeAllocatorType(), nullptr};
    kd_tree_data.Read(stream);

    return kd_tree_data;
  }

  static void Save(KdTreeData const& data, internal::Stream* stream) {
    // Write sdim.
    stream->Write(data.root_box.size());
    data.Write(stream);
  }

  //! \brief Sorted indices that refer to points inside points_.
  std::vector<IndexType> indices;
  //! \brief Bounding box of the root node.
  BoxType root_box;
  //! \brief Memory allocator for tree nodes.
  NodeAllocatorType allocator;
  //! \brief Root of the KdTree.
  NodeType* root_node;
};

//! \brief This class provides the build algorithm of the KdTree. How the KdTree
//! will be build depends on the Splitter template argument.
template <typename Traits_, typename Splitter_, Size Dim_, typename KdTreeData_>
class KdTreeBuilder {
 public:
  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  using SpaceType = typename Traits_::SpaceType;
  using BoxType = Box<ScalarType, Dim_>;
  using SplitterType = Splitter_;
  using KdTreeDataType = KdTreeData_;
  using NodeType = typename KdTreeDataType::NodeType;
  using NodeAllocatorType = typename KdTreeDataType::NodeAllocatorType;

  //! \brief Construct a KdTree given \p points , \p max_leaf_size and
  //! SplitterType.
  inline static KdTreeDataType Build(
      SpaceType const& points, IndexType const max_leaf_size) {
    assert(Traits_::SpaceNpts(points) > 0);
    assert(max_leaf_size > 0);

    std::vector<IndexType> indices(Traits_::SpaceNpts(points));
    std::iota(indices.begin(), indices.end(), 0);
    BoxType root_box = ComputeBoundingBox(points);
    NodeAllocatorType allocator;
    NodeType* root_node =
        KdTreeBuilder{points, max_leaf_size, &indices, &allocator}(root_box);

    return KdTreeDataType{
        std::move(indices), root_box, std::move(allocator), root_node};
  }

 private:
  inline KdTreeBuilder(
      SpaceType const& space,
      IndexType const max_leaf_size,
      std::vector<IndexType>* indices,
      NodeAllocatorType* allocator)
      : space_(space),
        max_leaf_size_{max_leaf_size},
        splitter_(space),
        indices_(*indices),
        allocator_{*allocator} {}

  //! \brief Creates the full set of nodes for a KdTree.
  inline NodeType* operator()(BoxType const& root_box) {
    BoxType box(root_box);
    return SplitIndices(0, indices_.begin(), indices_.end(), &box);
  }

  inline static BoxType ComputeBoundingBox(SpaceType const& points) {
    BoxType box(Traits_::SpaceSdim(points));
    box.FillInverseMax();
    for (IndexType i = 0; i < Traits_::SpaceNpts(points); ++i) {
      box.Fit(Traits_::PointCoords(Traits_::PointAt(points, i)));
    }
    return box;
  }

  //! \brief Creates a tree node for a range of indices, splits the range in two
  //! and recursively does the same for each sub set of indices until the index
  //! range size is less than or equal to max_leaf_size_.
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
      BoxType* box) const {
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
      splitter_(depth, begin, end, *box, &split, &split_dim, &split_val);

      BoxType right = *box;
      // Argument box will function as the left bounding box until we merge left
      // and right again at the end of this code section.
      box->max(split_dim) = split_val;
      right.min(split_dim) = split_val;

      node->left = SplitIndices(depth + 1, begin, split, box);
      node->right = SplitIndices(depth + 1, split, end, &right);

      SetBranch(*box, right, split_dim, node);

      // Merges both child boxes. We can expect any of the min max values to
      // change except for the ones of split_dim.
      box->Fit(right);
    }

    return node;
  }

  template <typename RandomAccessIterator_>
  inline void ComputeBoundingBox(
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType* box) const {
    box->FillInverseMax();
    for (; begin < end; ++begin) {
      box->Fit(Traits_::PointCoords(Traits_::PointAt(space_, *begin)));
    }
  }

  inline void SetBranch(
      BoxType const& left,
      BoxType const& right,
      SizeType const split_dim,
      KdTreeNodeEuclidean<IndexType, ScalarType>* node) const {
    node->data.branch.split_dim = static_cast<int>(split_dim);
    node->data.branch.left_max = left.max(split_dim);
    node->data.branch.right_min = right.min(split_dim);
  }

  inline void SetBranch(
      BoxType const& left,
      BoxType const& right,
      SizeType const split_dim,
      KdTreeNodeTopological<IndexType, ScalarType>* node) const {
    node->data.branch.split_dim = static_cast<int>(split_dim);
    node->data.branch.left_min = left.min(split_dim);
    node->data.branch.left_max = left.max(split_dim);
    node->data.branch.right_min = right.min(split_dim);
    node->data.branch.right_max = right.max(split_dim);
  }

  SpaceType const& space_;
  typename std::vector<IndexType>::difference_type const max_leaf_size_;
  SplitterType splitter_;
  std::vector<IndexType>& indices_;
  NodeAllocatorType& allocator_;
};

//! \brief This class provides a search nearest function for Euclidean spaces.
template <
    typename Traits_,
    typename Metric_,
    Size Dim_,
    typename Point_,
    typename Visitor_>
class SearchNearestEuclidean {
 public:
  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SpaceType = typename Traits_::SpaceType;
  //! \brief Node type supported by this SearchNearestEuclidean.
  using NodeType = KdTreeNodeEuclidean<IndexType, ScalarType>;

  inline SearchNearestEuclidean(
      SpaceType const& points,
      Metric_ const& metric,
      std::vector<IndexType> const& indices,
      Point_ const& query,
      Visitor_* visitor)
      : points_(points),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(Traits_::SpaceSdim(points)),
        visitor_(*visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) {
    node_box_offset_.Fill(ScalarType(0.0));
    SearchNearest(node, ScalarType(0.0));
  }

 private:
  inline void SearchNearest(
      NodeType const* const node, ScalarType node_box_distance) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        ScalarType const d =
            metric_(query_, Traits_::PointAt(points_, indices_[i]));
        if (visitor_.max() > d) {
          visitor_(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      ScalarType const v =
          Traits_::PointCoords(query_)[node->data.branch.split_dim];
      ScalarType new_offset;
      NodeType const* node_1st;
      NodeType const* node_2nd;

      // On equals we would possibly need to go left as well. However, this is
      // handled by the if statement below this one: the check that max search
      // radius still hits the split value after having traversed the first
      // branch.
      // If left_max - v > 0, this means that the query is inside the left node,
      // if right_min - v < 0 it's inside the right one. For the area in between
      // we just pick the closest one by summing them.
      if ((node->data.branch.left_max + node->data.branch.right_min - v - v) >
          0) {
        node_1st = node->left;
        node_2nd = node->right;
        new_offset = metric_(node->data.branch.right_min, v);
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = metric_(node->data.branch.left_max, v);
      }

      // S. Arya and D. M. Mount. Algorithms for fast vector quantization. In
      // IEEE Data Compression Conference, pages 381–390, March 1993
      // https://www.cs.umd.edu/~mount/Papers/DCC.pdf
      // This paper describes the "Incremental Distance Calculation" technique
      // to speed up nearest neighbor queries.

      // The distance and offset for node_1st is the same as that of its parent.
      SearchNearest(node_1st, node_box_distance);

      // Calculate the distance to node_2nd.
      // NOTE: This method only works with Lp norms to which the exponent is not
      // applied.
      ScalarType const old_offset =
          node_box_offset_[node->data.branch.split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[node->data.branch.split_dim] = new_offset;
        SearchNearest(node_2nd, node_box_distance);
        node_box_offset_[node->data.branch.split_dim] = old_offset;
      }
    }
  }

  SpaceType const& points_;
  Metric_ const& metric_;
  std::vector<IndexType> const& indices_;
  Point_ const& query_;
  Point<ScalarType, Dim_> node_box_offset_;
  Visitor_& visitor_;
};

//! \brief This class provides a search nearest function for topological spaces.
template <
    typename Traits_,
    typename Metric_,
    Size Dim_,
    typename Point_,
    typename Visitor_>
class SearchNearestTopological {
 public:
  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SpaceType = typename Traits_::SpaceType;
  //! \brief Node type supported by this SearchNearestTopological.
  using NodeType = KdTreeNodeTopological<IndexType, ScalarType>;

  inline SearchNearestTopological(
      SpaceType const& points,
      Metric_ const& metric,
      std::vector<IndexType> const& indices,
      Point_ const& query,
      Visitor_* visitor)
      : points_(points),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(Traits_::SpaceSdim(points)),
        visitor_(*visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) {
    node_box_offset_.Fill(ScalarType(0.0));
    SearchNearest(node, ScalarType(0.0));
  }

 private:
  inline void SearchNearest(
      NodeType const* const node, ScalarType node_box_distance) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        ScalarType const d =
            metric_(query_, Traits_::PointAt(points_, indices_[i]));
        if (visitor_.max() > d) {
          visitor_(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      ScalarType const v =
          Traits_::PointCoords(query_)[node->data.branch.split_dim];
      // Determine the distance to the boxes of the children of this node.
      ScalarType const d1 = metric_(
          v,
          node->data.branch.left_min,
          node->data.branch.left_max,
          node->data.branch.split_dim);
      ScalarType const d2 = metric_(
          v,
          node->data.branch.right_min,
          node->data.branch.right_max,
          node->data.branch.split_dim);
      NodeType const* node_1st;
      NodeType const* node_2nd;
      ScalarType new_offset;

      // Visit the closest child/box first.
      if (d1 < d2) {
        node_1st = node->left;
        node_2nd = node->right;
        new_offset = d2;
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = d1;
      }

      SearchNearest(node_1st, node_box_distance);

      ScalarType const old_offset =
          node_box_offset_[node->data.branch.split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[node->data.branch.split_dim] = new_offset;
        SearchNearest(node_2nd, node_box_distance);
        node_box_offset_[node->data.branch.split_dim] = old_offset;
      }
    }
  }

  SpaceType const& points_;
  Metric_ const& metric_;
  std::vector<IndexType> const& indices_;
  Point_ const& query_;
  Point<ScalarType, Dim_> node_box_offset_;
  Visitor_& visitor_;
};

//! \brief A functor that provides range searches for Euclidean spaces. Query
//! time is bounded by O(n^(1-1/Dim)+k).
//! \details Many tree nodes are excluded by checking if they intersect with the
//! box of the query. We don't store the bounding box of each node but calculate
//! them at run time. This slows down SearchBox in favor of having faster
//! nearest neighbor searches.
template <typename Traits_, typename Metric_, Size Dim_>
class SearchBoxEuclidean {
 public:
  // TODO Perhaps we can support it for both topological and Euclidean spaces.
  static_assert(
      std::is_same<typename Metric_::SpaceTag, EuclideanSpaceTag>::value,
      "SEARCH_BOX_ONLY_SUPPORTED_FOR_EUCLIDEAN_SPACES");

  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SpaceType = typename Traits_::SpaceType;

  inline SearchBoxEuclidean(
      SpaceType const& points,
      Metric_ const& metric,
      std::vector<IndexType> const& indices,
      Box<ScalarType, Dim_> const& root_box,
      BoxMap<ScalarType const, Dim_> const& query,
      std::vector<IndexType>* idxs)
      : points_(points),
        metric_(metric),
        indices_(indices),
        box_(root_box),
        query_(query),
        idxs_(*idxs) {}

  //! \brief Range search starting from \p node.
  template <typename Node>
  inline void operator()(Node const* const node) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        if (query_.Contains(
                Traits_::PointCoords(Traits_::PointAt(points_, indices_[i])))) {
          idxs_.push_back(indices_[i]);
        }
      }
    } else {
      ScalarType old_value = box_.max(node->data.branch.split_dim);
      box_.max(node->data.branch.split_dim) = node->data.branch.left_max;

      // Check if the left node is fully contained. If true, report all its
      // indices. Else, if its partially contained, continue the range search
      // down the left node.
      if (query_.Contains(box_)) {
        ReportNode(node->left);
      } else if (
          query_.min(node->data.branch.split_dim) <
          node->data.branch.left_max) {
        operator()(node->left);
      }

      box_.max(node->data.branch.split_dim) = old_value;
      old_value = box_.min(node->data.branch.split_dim);
      box_.min(node->data.branch.split_dim) = node->data.branch.right_min;

      // Same as the left side.
      if (query_.Contains(box_)) {
        ReportNode(node->right);
      } else if (
          query_.max(node->data.branch.split_dim) >
          node->data.branch.right_min) {
        operator()(node->right);
      }

      box_.min(node->data.branch.split_dim) = old_value;
    }
  }

 private:
  //! \brief Reports all indices contained by \p node.
  template <typename Node>
  inline void ReportNode(Node const* const node) const {
    if (node->IsLeaf()) {
      std::copy(
          indices_.cbegin() + node->data.leaf.begin_idx,
          indices_.cbegin() + node->data.leaf.end_idx,
          std::back_inserter(idxs_));
    } else {
      ReportNode(node->left);
      ReportNode(node->right);
    }
  }

  SpaceType const& points_;
  Metric_ const& metric_;
  std::vector<IndexType> const& indices_;
  // This variable is used for maintaining a running bounding box.
  Box<ScalarType, Dim_> box_;
  BoxMap<ScalarType const, Dim_> const& query_;
  std::vector<IndexType>& idxs_;
};

}  // namespace internal

//! \brief Splits a node on the median of the longest dimension of its box. Also
//! known as the standard split rule.
//! \details This splitting build a tree in O(n log n) time on average. It's
//! generally slower compared to SplitterSlidingMidpoint but results in a
//! balanced KdTree.
template <typename Traits_>
class SplitterLongestMedian {
 public:
  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  using SpaceType = typename Traits_::SpaceType;
  template <SizeType Dim_>
  using BoxType = typename internal::Box<ScalarType, Dim_>;

  SplitterLongestMedian(SpaceType const& points) : points_{points} {}

  //! \brief This function splits a node.
  template <typename RandomAccessIterator_, SizeType Dim_>
  inline void operator()(
      IndexType const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType<Dim_> const& box,
      RandomAccessIterator_* split,
      SizeType* split_dim,
      ScalarType* split_val) const {
    ScalarType max_delta;
    box.LongestAxis(split_dim, &max_delta);

    *split = begin + (end - begin) / 2;

    std::nth_element(
        begin,
        *split,
        end,
        [this, &split_dim](IndexType const a, IndexType const b) -> bool {
          return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
        });

    *split_val = PointCoord(**split, *split_dim);
  }

 private:
  inline ScalarType const& PointCoord(
      IndexType point_idx, SizeType coord_idx) const {
    return Traits_::PointCoords(
        Traits_::PointAt(points_, point_idx))[coord_idx];
  }

  SpaceType const& points_;
};

//! \brief Bounding boxes of tree nodes are split in the middle along the
//! longest axis unless this results in an empty sub-node. In this case the
//! split gets adjusted to fit a single point into this sub-node.
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
//! faster to build and query as compared to SplitterLongestMedian.
template <typename Traits_>
class SplitterSlidingMidpoint {
 public:
  using IndexType = typename Traits_::IndexType;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  using SpaceType = typename Traits_::SpaceType;
  template <SizeType Dim_>
  using BoxType = typename internal::Box<ScalarType, Dim_>;

  SplitterSlidingMidpoint(SpaceType const& points) : points_{points} {}

  //! \brief This function splits a node.
  template <typename RandomAccessIterator_, SizeType Dim_>
  inline void operator()(
      IndexType const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      BoxType<Dim_> const& box,
      RandomAccessIterator_* split,
      SizeType* split_dim,
      ScalarType* split_val) const {
    ScalarType max_delta;
    box.LongestAxis(split_dim, &max_delta);
    *split_val = max_delta / ScalarType(2.0) + box.min(*split_dim);

    // Everything smaller than split_val goes left, the rest right.
    auto const comp =
        [this, &split_dim, &split_val](IndexType const a) -> bool {
      return PointCoord(a, *split_dim) < *split_val;
    };

    *split = std::partition(begin, end, comp);

    // If it happens that either all points are on the left side or right side,
    // one point slides to the other side and we split on the first right value
    // instead of the middle split.
    // In these two cases the split value is unknown and a partial sort is
    // required to obtain it, but also to rearrange all other indices such that
    // they are on their corresponding left or right side.
    if ((*split) == end) {
      (*split)--;
      std::nth_element(
          begin,
          *split,
          end,
          [this, &split_dim](IndexType const a, IndexType const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(**split, *split_dim);
    } else if ((*split) == begin) {
      (*split)++;
      std::nth_element(
          begin,
          *split,
          end,
          [this, &split_dim](IndexType const a, IndexType const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(**split, *split_dim);
    }
  }

 private:
  inline ScalarType const& PointCoord(
      IndexType point_idx, SizeType coord_idx) const {
    return Traits_::PointCoords(
        Traits_::PointAt(points_, point_idx))[coord_idx];
  }

  SpaceType const& points_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dim_ The spatial dimension of the tree. Dim_ defaults to
//! Traits_::Dim but can be set to a different value should a rare occasion
//! require it.
template <
    typename Traits_,
    typename Metric_ = L2Squared<Traits_>,
    typename Splitter_ = SplitterSlidingMidpoint<Traits_>,
    Size Dim_ = Traits_::Dim>
class KdTree {
 private:
  // 1) Dim_ <= Traits_::Dim_. The spatial dimension represented by the tree can
  // be smaller than that of the input data.
  // 2) Dim_ == kDynamicDim. The spatial dimension will be determined run-time
  // from the traits.
  // 3) Dim_ != kDynamicDim && Traits_::Dim_ == kDynamicDim. As handled by 1),
  // but here it's made explicit that while the traits represent a dynamic
  // dimension, the tree can take on a static dimension.
  static_assert(
      (Dim_ <= Traits_::Dim || Dim_ == kDynamicDim) && Dim_ > 0,
      "SPATIAL_DIMENSION_TREE_MUST_BE_SMALLER_OR_EQUAL_TO_TRAITS_DIMENSION");

 public:
  //! \brief Index type.
  using IndexType = typename Traits_::IndexType;
  //! \brief Scalar type.
  using ScalarType = typename Traits_::ScalarType;
  //! \brief KdTree dimension. It equals pico_tree::kDynamicDim in case Dim is
  //! only known at run-time.
  static Size constexpr Dim = Dim_;
  //! \brief Traits_ with information about the input Spaces and Points.
  using TraitsType = Traits_;
  //! \brief Point set or adaptor type.
  using SpaceType = typename Traits_::SpaceType;
  //! \brief The metric used for various searches.
  using MetricType = Metric_;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<IndexType, ScalarType>;

 private:
  //! \brief Node type based on Metric_::SpaceTag.
  using NodeType = typename internal::KdTreeSpaceTagTraits<
      typename Metric_::SpaceTag>::template NodeType<IndexType, ScalarType>;
  using KdTreeDataType =
      internal::KdTreeData<IndexType, ScalarType, Dim_, NodeType>;

 public:
  //! \brief Creates a KdTree given \p points and \p max_leaf_size.
  //!
  //! \details
  //! The KdTree wants ownership of \p points to avoid problems that may occur
  //! when keeping a const reference to the data. For example, using std::move()
  //! on the point set would invalidate the local reference which is then
  //! unrecoverable.
  //!
  //! To avoid a deep copy of the \p points object:
  //! \li Move it: KdTree tree(std::move(points), max_leaf_size);
  //! \li Implement its class as an adaptor that keeps a reference to the data.
  //!
  //! The value of \p max_leaf_size influences the height and performance of the
  //! tree. The splitting mechanism determines data locality within the leafs.
  //! The exact effect it has depends on the tree splitting mechanism.
  //!
  //! \param points The input point set (interface).
  //! \param max_leaf_size The maximum amount of points allowed in a leaf node.
  KdTree(SpaceType points, IndexType max_leaf_size)
      : points_(std::move(points)),
        metric_(),
        data_(
            internal::KdTreeBuilder<Traits_, Splitter_, Dim_, KdTreeDataType>::
                Build(points_, max_leaf_size)) {}

  //! \brief The KdTree cannot be copied.
  //! \details The KdTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  KdTree(KdTree const&) = delete;

  //! \brief Move constructor of the KdTree.
  KdTree(KdTree&&) = default;

  //! \brief KdTree copy assignment.
  KdTree& operator=(KdTree const& other) = delete;

  //! \brief KdTree move assignment.
  KdTree& operator=(KdTree&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNearest(P const& x, V* visitor) const {
    SearchNearest(data_.root_node, x, visitor, typename Metric_::SpaceTag());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default L2Squared results in a squared distance.
  template <typename P>
  inline void SearchNn(P const& x, NeighborType* nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(x, &v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<IndexType, ScalarType>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \tparam RandomAccessIterator Iterator type.
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& x, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchKnn<RandomAccessIterator> v(begin, end);
    SearchNearest(x, &v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x, IndexType const k, std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits_::SpaceNpts(points_)));
    SearchKnn(x, knn->begin(), knn->end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! Metric. The default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \param x Input point.
  //! \param radius Search radius.
  //! \code{.cpp}
  //! ScalarType distance = -2.0;
  //! // E.g., L1: 2.0, L2Squared: 4.0
  //! ScalarType metric_distance = kdtree.metric()(distance);
  //! std::vector<Neighbor<IndexType, ScalarType>> n;
  //! tree.SearchRadius(p, metric_distance, &n);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& x,
      ScalarType const radius,
      std::vector<NeighborType>* n,
      bool const sort = false) const {
    internal::SearchRadius<NeighborType> v(radius, n);
    SearchNearest(x, &v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p x,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals Neighbor<IndexType, ScalarType>.
  //! \details This function can result in faster search queries compared to
  //! KdTree::SearchKnn by skipping points and tree nodes. This is achieved by
  //! scaling down the search distance, possibly not visiting the true nearest
  //! neighbor. An approximate nearest neighbor will at most be a factor of
  //! distance ratio \p e farther from the query point than the true nearest
  //! neighbor: max_ann_distance = true_nn_distance * e. This holds true for
  //! each respective nn index i, 0 <= i < k.
  //!
  //! The amount of requested neighbors, k, should be sufficiently large to get
  //! a noticeable speed increase from this method. Within a leaf all points are
  //! compared to the query anyway, even if they are skipped. These calculations
  //! can be avoided by skipping leafs completely, which will never happen if
  //! all requested neighbors reside within a single one.
  //!
  //! Interpretation of both the input error ratio and output distances
  //! depend on the Metric. The default L2Squared calculates squared
  //! distances. Using this metric, the input error ratio should be the squared
  //! error ratio and the output distances will be squared distances scaled by
  //! the inverse error ratio.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! ScalarType max_error = ScalarType(0.15);
  //! ScalarType e = tree.metric()(ScalarType(1.0) + max_error);
  //! std::vector<Neighbor<IndexType, ScalarType>> knn(k);
  //! tree.SearchAknn(p, e, knn.begin(), knn.end());
  //! // Optionally scale back to the actual metric distance.
  //! for (auto& nn : knn) { nn.second *= e; }
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchAknn(
      P const& x,
      ScalarType const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNearest(x, &v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchAknn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchAknn(
      P const& x,
      IndexType const k,
      ScalarType const e,
      std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits_::SpaceNpts(points_)));
    SearchAknn(x, e, knn->begin(), knn->end());
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dim)+k).
  //! \tparam P Point type.
  template <typename P>
  inline void SearchBox(
      P const& min, P const& max, std::vector<IndexType>* idxs) const {
    idxs->clear();
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there is any
    // overlap then the search is slower. So unless many queries don't intersect
    // there is no point in adding it.
    internal::SearchBoxEuclidean<Traits_, Metric_, Dim>(
        points_,
        metric_,
        data_.indices,
        data_.root_box,
        internal::BoxMap<ScalarType const, Dim>(
            Traits_::PointCoords(min),
            Traits_::PointCoords(max),
            Traits_::SpaceSdim(points_)),
        idxs)(data_.root_node);
  }

  //! \brief Point set used by the tree.
  inline SpaceType const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline MetricType const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static KdTree Load(SpaceType points, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::in | std::ios::binary);
    return Load(std::move(points), &stream);
  }

  //! \brief Loads the tree in binary from \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Does not check if the stored tree structure is valid for the given
  //! point set.
  //! \li Does not check if the stored tree structure is valid for the given
  //! template arguments.
  static KdTree Load(SpaceType points, std::iostream* stream) {
    internal::Stream s(stream);
    return KdTree(std::move(points), &s);
  }

  //! \brief Saves the tree in binary to file.
  static void Save(KdTree const& tree, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::out | std::ios::binary);
    Save(tree, &stream);
  }

  //! \brief Saves the tree in binary to \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Stores the tree structure but not the points.
  static void Save(KdTree const& tree, std::iostream* stream) {
    internal::Stream s(stream);
    KdTreeDataType::Save(tree.data_, &s);
  }

 private:
  //! \brief Constructs a KdTree by reading its indexing and leaf information
  //! from a Stream.
  KdTree(SpaceType points, internal::Stream* stream)
      : points_(std::move(points)),
        metric_(),
        data_(KdTreeDataType::Load(stream)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename P, typename V>
  inline void SearchNearest(
      NodeType const* const node,
      P const& x,
      V* visitor,
      EuclideanSpaceTag) const {
    internal::SearchNearestEuclidean<Traits_, Metric_, Dim, P, V>(
        points_, metric_, data_.indices, x, visitor)(node);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename P, typename V>
  inline void SearchNearest(
      NodeType const* const node,
      P const& x,
      V* visitor,
      TopologicalSpaceTag) const {
    internal::SearchNearestTopological<Traits_, Metric_, Dim, P, V>(
        points_, metric_, data_.indices, x, visitor)(node);
  }

  //! \brief Point set adapter used for querying point data.
  SpaceType points_;
  //! \brief Metric used for comparing distances.
  MetricType metric_;
  //! \brief Data structure of the KdTree.
  KdTreeDataType data_;
};

}  // namespace pico_tree
