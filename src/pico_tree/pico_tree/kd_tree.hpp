#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/sequence.hpp"
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
template <typename Index>
struct KdTreeLeaf {
  //! \private
  Index begin_idx;
  //! \private
  Index end_idx;
};

//! \brief Tree branch.
template <typename Scalar>
struct KdTreeBranchSplit {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar right_min;
};

//! \brief Tree branch.
//! \details This branch version allows identifications (wrapping around) by
//! storing the boundaries of the box that corresponds to the current node. The
//! split value allows arbitrary splitting techniques.
template <typename Scalar>
struct KdTreeBranchRange {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Minimum coordinate value of the left node box for split_dim.
  Scalar left_min;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar right_min;
  //! \brief Maximum coordinate value of the right node box for split_dim.
  Scalar right_max;
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
template <typename Index, typename Scalar>
struct KdTreeNodeEuclidean
    : public KdTreeNodeBase<KdTreeNodeEuclidean<Index, Scalar>> {
  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index>, KdTreeBranchSplit<Scalar>> data;
};

//! \brief KdTree node for a topological space.
template <typename Index, typename Scalar>
struct KdTreeNodeTopological
    : public KdTreeNodeBase<KdTreeNodeTopological<Index, Scalar>> {
  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index>, KdTreeBranchRange<Scalar>> data;
};

//! \brief KdTree meta information depending on the SpaceTag template argument.
template <typename SpaceTag>
struct KdTreeSpaceTagTraits;

//! \brief KdTree meta information for the EuclideanSpaceTag.
template <>
struct KdTreeSpaceTagTraits<EuclideanSpaceTag> {
  //! \brief Supported node type.
  template <typename Index, typename Scalar>
  using Node = KdTreeNodeEuclidean<Index, Scalar>;
};

//! \brief KdTree meta information for the TopologicalSpaceTag.
template <>
struct KdTreeSpaceTagTraits<TopologicalSpaceTag> {
  //! \brief Supported node type.
  template <typename Index, typename Scalar>
  using Node = KdTreeNodeTopological<Index, Scalar>;
};

//! \brief A SequenceBox can be used as a bounding box. It uses a Sequence for
//! storing the min and max coordinate of the box.
template <typename Scalar_, int Dim_>
struct SequenceBox {
  //! \brief Minimum box coordinate.
  Sequence<Scalar_, Dim_> min;
  //! \brief Maximum box coordinate.
  Sequence<Scalar_, Dim_> max;
};

//! \brief This class provides the build algorithm of the KdTree. How the KdTree
//! is build depends on the Splitter template argument.
template <typename Traits, typename SpaceTag, typename Splitter, int Dim_>
class KdTreeBuilder {
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  using SequenceBoxType = SequenceBox<Scalar, Dim_>;

 public:
  //! \brief Node type supported by KdTreeBuilder based on SpaceTag.
  using Node =
      typename KdTreeSpaceTagTraits<SpaceTag>::template Node<Index, Scalar>;
  //! \brief Memory buffer type supported by KdTreeBuilder based on SpaceTag.
  using MemoryBuffer = typename Splitter::template MemoryBuffer<Node>;

  //! \brief Constructs a KdTreeBuilder.
  inline KdTreeBuilder(
      Space const& space,
      std::vector<Index> const& indices,
      Index const max_leaf_size,
      Splitter const& splitter,
      MemoryBuffer* nodes)
      : space_(space),
        indices_(indices),
        max_leaf_size_{max_leaf_size},
        splitter_{splitter},
        nodes_{*nodes} {}

  //! \brief Creates the full set of nodes for a KdTree.
  inline Node* operator()(SequenceBoxType const& root_box) const {
    SequenceBoxType box(root_box);
    return SplitIndices(0, 0, Traits::SpaceNpts(space_), &box);
  }

 private:
  //! \brief Creates a tree node for a range of indices, splits the range in two
  //! and recursively does the same for each sub set of indices until the index
  //! range \p size is less than or equal to max_leaf_size_.
  //! \details While descending the tree we split nodes based on the root box
  //! until leaf nodes are reached. Inside the leaf nodes the boxes are updated
  //! to be the bounding boxes of the points they contain. While unwinding the
  //! recursion we update the split information for each branch node based on
  //! merging leaf nodes. Since the updated split informaton based on the leaf
  //! nodes can have smaller bounding boxes than the original ones, we can
  //! improve query times.
  inline Node* SplitIndices(
      Index const depth,
      Index const offset,
      Index const size,
      SequenceBoxType* box) const {
    Node* node = nodes_.Allocate();
    //
    if (size <= max_leaf_size_) {
      node->data.leaf.begin_idx = offset;
      node->data.leaf.end_idx = offset + size;
      node->left = nullptr;
      node->right = nullptr;
      // Keep the original box in case it was empty.
      if (size > 0) {
        CalculateBoundingBox(
            node->data.leaf.begin_idx, node->data.leaf.end_idx, box);
      }
    } else {
      int split_dim;
      Index split_idx;
      Scalar split_val;
      splitter_(
          depth,
          offset,
          size,
          box->min,
          box->max,
          &split_dim,
          &split_idx,
          &split_val);

      // The split_idx is used as the first index of the right branch.
      Index const left_size = split_idx - offset;
      Index const right_size = size - left_size;

      SequenceBoxType right = *box;
      // Argument box will function as the left bounding box until we merge left
      // and right again at the end of this code section.
      box->max[split_dim] = split_val;
      right.min[split_dim] = split_val;

      node->left = SplitIndices(depth + 1, offset, left_size, box);
      node->right = SplitIndices(depth + 1, split_idx, right_size, &right);

      SetBranch(*box, right, split_dim, node);

      // This loop merges both child boxes. We can expect any of the min max
      // values to change except for the one of split_dim.
      for (int i = 0; i < Dimension<Traits, Dim_>::Dim(space_); ++i) {
        if (right.min[i] < box->min[i]) {
          box->min[i] = right.min[i];
        }

        if (right.max[i] > box->max[i]) {
          box->max[i] = right.max[i];
        }
      }
    }

    return node;
  }

  inline void CalculateBoundingBox(
      Index const begin_idx, Index const end_idx, SequenceBoxType* box) const {
    box->min.Fill(
        Traits::SpaceSdim(space_), std::numeric_limits<Scalar>::max());
    box->max.Fill(
        Traits::SpaceSdim(space_), std::numeric_limits<Scalar>::lowest());

    for (Index j = begin_idx; j < end_idx; ++j) {
      Scalar const* const p =
          Traits::PointCoords(Traits::PointAt(space_, indices_[j]));
      for (int i = 0; i < Dimension<Traits, Dim_>::Dim(space_); ++i) {
        Scalar const v = p[i];
        if (v < box->min[i]) {
          box->min[i] = v;
        }

        if (v > box->max[i]) {
          box->max[i] = v;
        }
      }
    }
  }

  inline void SetBranch(
      SequenceBoxType const& left,
      SequenceBoxType const& right,
      int const split_dim,
      KdTreeNodeEuclidean<Index, Scalar>* node) const {
    node->data.branch.split_dim = split_dim;
    node->data.branch.left_max = left.max[split_dim];
    node->data.branch.right_min = right.min[split_dim];
  }

  inline void SetBranch(
      SequenceBoxType const& left,
      SequenceBoxType const& right,
      int const split_dim,
      KdTreeNodeTopological<Index, Scalar>* node) const {
    node->data.branch.split_dim = split_dim;
    node->data.branch.left_min = left.min[split_dim];
    node->data.branch.left_max = left.max[split_dim];
    node->data.branch.right_min = right.min[split_dim];
    node->data.branch.right_max = right.max[split_dim];
  }

  Space const& space_;
  std::vector<Index> const& indices_;
  Index const max_leaf_size_;
  Splitter const& splitter_;
  MemoryBuffer& nodes_;
};

//! \brief This class provides a search nearest function for Euclidean spaces.
template <
    typename Traits,
    typename Metric,
    int Dim_,
    typename Point,
    typename Visitor>
class SearchNearestEuclidean {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;

 public:
  //! \brief Node type supported by this SearchNearestEuclidean.
  using Node = KdTreeNodeEuclidean<Index, Scalar>;

  //! \private
  inline SearchNearestEuclidean(
      Space const& points,
      Metric const& metric,
      std::vector<Index> const& indices,
      Point const& point,
      Visitor* visitor)
      : points_(points),
        metric_(metric),
        indices_(indices),
        point_(point),
        visitor_(*visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(Node const* const node) {
    node_box_offset_.Fill(Traits::SpaceSdim(points_), Scalar(0.0));
    SearchNearest(node, Scalar(0.0));
  }

 private:
  //! \private
  inline void SearchNearest(Node const* const node, Scalar node_box_distance) {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Scalar const d = metric_(point_, Traits::PointAt(points_, indices_[i]));
        if (visitor_.max() > d) {
          visitor_(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      Scalar const v = Traits::PointCoords(point_)[node->data.branch.split_dim];
      Scalar new_offset;
      Node const* node_1st;
      Node const* node_2nd;

      // On equals we would possibly need to go left as well. However, this is
      // handled by the if statement below this one: the check that max search
      // radius still hits the split value after having traversed the first
      // branch.
      // If left_max - v > 0, this means that the point is inside the left node,
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
      Scalar const old_offset = node_box_offset_[node->data.branch.split_dim];
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

  Space const& points_;
  Metric const& metric_;
  std::vector<Index> const& indices_;
  Point const& point_;
  Sequence<Scalar, Dim_> node_box_offset_;
  Visitor& visitor_;
};

//! \brief This class provides a search nearest function for topological spaces.
template <
    typename Traits,
    typename Metric,
    int Dim_,
    typename Point,
    typename Visitor>
class SearchNearestTopological {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;

 public:
  //! \brief Node type supported by this SearchNearestTopological.
  using Node = KdTreeNodeTopological<Index, Scalar>;

  //! \private
  inline SearchNearestTopological(
      Space const& points,
      Metric const& metric,
      std::vector<Index> const& indices,
      Point const& point,
      Visitor* visitor)
      : points_(points),
        metric_(metric),
        indices_(indices),
        point_(point),
        visitor_(*visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(Node const* const node) {
    node_box_offset_.Fill(Traits::SpaceSdim(points_), Scalar(0.0));
    SearchNearest(node, Scalar(0.0));
  }

 private:
  inline void SearchNearest(Node const* const node, Scalar node_box_distance) {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Scalar const d = metric_(point_, Traits::PointAt(points_, indices_[i]));
        if (visitor_.max() > d) {
          visitor_(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      Scalar const v = Traits::PointCoords(point_)[node->data.branch.split_dim];
      // Determine the distance to the boxes of the children of this node.
      Scalar const d1 = metric_(
          v,
          node->data.branch.left_min,
          node->data.branch.left_max,
          node->data.branch.split_dim);
      Scalar const d2 = metric_(
          v,
          node->data.branch.right_min,
          node->data.branch.right_max,
          node->data.branch.split_dim);
      Node const* node_1st;
      Node const* node_2nd;
      Scalar new_offset;

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

      Scalar const old_offset = node_box_offset_[node->data.branch.split_dim];
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

  Space const& points_;
  Metric const& metric_;
  std::vector<Index> const& indices_;
  Point const& point_;
  Sequence<Scalar, Dim_> node_box_offset_;
  Visitor& visitor_;
};

//! \brief This class provides the search box function.
template <typename Traits, typename Metric, int Dim_>
class SearchBox {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;

 public:
  //! \brief Returns all points within the box defined by \p rng_min and \p
  //! rng_max for \p node. Query time is bounded by O(n^(1-1/Dim)+k).
  //! \details Many tree nodes are excluded by checking if they intersect with
  //! the box of the query. We don't store the bounding box of each node but
  //! calculate them at run time. This slows down SearchBox in favor of having
  //! faster nearest neighbor searches.
  inline SearchBox(
      Space const& points,
      Metric const& metric,
      std::vector<Index> const& indices,
      Scalar const* const rng_min,
      Scalar const* const rng_max,
      std::vector<Index>* idxs)
      : points_(points),
        metric_(metric),
        indices_(indices),
        rng_min_(rng_min),
        rng_max_(rng_max),
        idxs_(*idxs) {}

  template <typename Node>
  inline void operator()(
      Node const* const node, SequenceBox<Scalar, Dim_>* box) const {
    // TODO Perhaps we can support it for both topological and Euclidean spaces.
    static_assert(
        std::is_same<typename Metric::SpaceTag, EuclideanSpaceTag>::value,
        "SEARCH_BOX_ONLY_SUPPORTED_FOR_EUCLIDEAN_SPACES");
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Index const idx = indices_[i];
        if (PointInBox(
                rng_min_,
                rng_max_,
                Traits::PointCoords(Traits::PointAt(points_, idx)))) {
          idxs_.push_back(idx);
        }
      }
    } else {
      Scalar old_value = box->max[node->data.branch.split_dim];
      box->max[node->data.branch.split_dim] = node->data.branch.left_max;

      // Check if the left node is fully contained. If true, report all its
      // indices. Else, if its partially contained, continue the range search
      // down the left node.
      if (PointInBox(rng_min_, rng_max_, box->min) &&
          PointInBox(rng_min_, rng_max_, box->max)) {
        ReportNode(node->left);
      } else if (
          rng_min_[node->data.branch.split_dim] < node->data.branch.left_max) {
        operator()(node->left, box);
      }

      box->max[node->data.branch.split_dim] = old_value;
      old_value = box->min[node->data.branch.split_dim];
      box->min[node->data.branch.split_dim] = node->data.branch.right_min;

      // Same as the left side.
      if (PointInBox(rng_min_, rng_max_, box->min) &&
          PointInBox(rng_min_, rng_max_, box->max)) {
        ReportNode(node->right);
      } else if (
          rng_max_[node->data.branch.split_dim] > node->data.branch.right_min) {
        operator()(node->right, box);
      }

      box->min[node->data.branch.split_dim] = old_value;
    }
  }

 private:
  //! Checks if \p x is contained in the box defined by \p min and \p max. A
  //! point on the edge considered inside the box.
  template <typename P0, typename P1>
  inline bool PointInBox(P0 const& min, P0 const& max, P1 const& x) const {
    for (int i = 0; i < internal::Dimension<Traits, Dim_>::Dim(points_); ++i) {
      if (min[i] > x[i] || max[i] < x[i]) {
        return false;
      }
    }
    return true;
  }

  //! Reports all indices contained by \p node.
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

  Space const& points_;
  Metric const& metric_;
  std::vector<Index> const& indices_;
  Scalar const* const rng_min_;
  Scalar const* const rng_max_;
  std::vector<Index>& idxs_;
};

//! \brief See which axis of the box is the longest.
template <typename Scalar, int Dim>
inline void LongestAxisBox(
    Sequence<Scalar, Dim> const& box_min,
    Sequence<Scalar, Dim> const& box_max,
    int* p_max_index,
    Scalar* p_max_value) {
  assert(box_min.size() == box_max.size());

  *p_max_value = std::numeric_limits<Scalar>::lowest();

  for (int i = 0; i < static_cast<int>(box_min.size()); ++i) {
    Scalar const delta = box_max[i] - box_min[i];
    if (delta > *p_max_value) {
      *p_max_index = i;
      *p_max_value = delta;
    }
  }
}

}  // namespace internal

//! \brief Splits a node on the median of the longest dimension of its box. Also
//! known as the standard split rule.
//! \details This splitting build a tree in O(n log n) time on average. It's
//! generally slower compared to SplitterSlidingMidpoint but results in a
//! balanced KdTree.
template <typename Traits>
class SplitterLongestMedian {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  template <int Dim_>
  using Sequence = typename internal::Sequence<Scalar, Dim_>;

 public:
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::StaticBuffer<T>;

  //! \private
  SplitterLongestMedian(Space const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  template <int Dim_>
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence<Dim_> const& box_min,
      Sequence<Dim_> const& box_max,
      int* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);

    *split_idx = size / 2 + offset;

    std::nth_element(
        indices_.begin() + offset,
        indices_.begin() + *split_idx,
        indices_.begin() + offset + size,
        [this, &split_dim](Index const a, Index const b) -> bool {
          return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
        });

    *split_val = PointCoord(indices_[*split_idx], *split_dim);
  }

 private:
  inline Scalar const& PointCoord(
      Index const point_idx, int const coord_idx) const {
    return Traits::PointCoords(Traits::PointAt(points_, point_idx))[coord_idx];
  }

  Space const& points_;
  std::vector<Index>& indices_;
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
template <typename Traits>
class SplitterSlidingMidpoint {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  template <int Dim_>
  using Sequence = typename internal::Sequence<Scalar, Dim_>;

 public:
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::DynamicBuffer<T>;

  //! \private
  SplitterSlidingMidpoint(Space const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  template <int Dim_>
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence<Dim_> const& box_min,
      Sequence<Dim_> const& box_max,
      int* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);
    *split_val = max_delta / Scalar(2.0) + box_min[*split_dim];

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](Index const a) -> bool {
      return PointCoord(a, *split_dim) < *split_val;
    };

    *split_idx = static_cast<Index>(
        std::partition(
            indices_.begin() + offset, indices_.begin() + offset + size, comp) -
        indices_.cbegin());

    // If it happens that either all points are on the left side or right side,
    // one point slides to the other side and we split on the first right value
    // instead of the middle split.
    // In these two cases the split value is unknown and a partial sort is
    // required to obtain it, but also to rearrange all other indices such that
    // they are on their corresponding left or right side.
    if ((*split_idx - offset) == size) {
      (*split_idx)--;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [this, &split_dim](Index const a, Index const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(indices_[*split_idx], *split_dim);
    } else if ((*split_idx - offset) == 0) {
      (*split_idx)++;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [this, &split_dim](Index const a, Index const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(indices_[*split_idx], *split_dim);
    }
  }

 private:
  inline Scalar const& PointCoord(Index point_idx, int coord_idx) const {
    return Traits::PointCoords(Traits::PointAt(points_, point_idx))[coord_idx];
  }

  Space const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dim_ The spatial dimension of the tree. Dim_ defaults to Traits::Dim
//! but can be set to a different value should a rare occasion require it.
template <
    typename Traits,
    typename Metric = L2Squared<Traits>,
    typename Splitter = SplitterSlidingMidpoint<Traits>,
    int Dim_ = Traits::Dim>
class KdTree {
 private:
  static_assert(
      Dim_ <= Traits::Dim,
      "SPATIAL_DIMENSION_TREE_MUST_BE_SMALLER_OR_EQUAL_TO_TRAITS_DIMENSION");

  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  using Builder = internal::
      KdTreeBuilder<Traits, typename Metric::SpaceTag, Splitter, Dim_>;
  using Node = typename Builder::Node;
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dim_>;
  using SequenceBox = typename internal::SequenceBox<Scalar, Dim_>;
  using MemoryBuffer = typename Splitter::template MemoryBuffer<Node>;

 public:
  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = Scalar;
  //! \brief KdTree dimension. It equals pico_tree::kDynamicDim in case Dim is
  //! only known at run-time.
  static constexpr int Dim = Dim_;
  //! \brief Traits with information about the input Spaces and Points.
  using TraitsType = Traits;
  //! \brief Point set or adaptor type.
  using SpaceType = Space;
  //! \brief The metric used for various searches.
  using MetricType = Metric;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<Index, Scalar>;

 public:
  //! \brief The KdTree cannot be copied.
  //! \details The KdTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy. For now we are not interested in
  //! providing a deep copy.
  //! \private
  KdTree(KdTree const&) = delete;

  //! \brief Move constructor of the KdTree.
  //! \details The move constructor is not implicitly created because of the
  //! deleted copy constructor.
  //! \private
  KdTree(KdTree&&) = default;

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
  KdTree(Space points, Index max_leaf_size)
      : points_(std::move(points)),
        metric_(),
        nodes_(internal::MaxNodesFromPoints(Traits::SpaceNpts(points_))),
        indices_(Traits::SpaceNpts(points_)),
        root_(Build(max_leaf_size)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNearest(P const& x, V* visitor) const {
    SearchNearest(root_, x, visitor, typename Metric::SpaceTag());
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
  //! iterator equals Neighbor<Index, Scalar>.
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
      P const& x, Index const k, std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits::SpaceNpts(points_)));
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
  //! Scalar distance = -2.0;
  //! // E.g., L1: 2.0, L2Squared: 4.0
  //! Scalar metric_distance = kdtree.metric()(distance);
  //! std::vector<Neighbor<Index, Scalar>> n;
  //! tree.SearchRadius(p, metric_distance, &n);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& x,
      Scalar const radius,
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
  //! type of the iterator equals Neighbor<Index, Scalar>.
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
  //! Scalar max_error = Scalar(0.15);
  //! Scalar e = tree.metric()(Scalar(1.0) + max_error);
  //! std::vector<Neighbor<Index, Scalar>> knn(k);
  //! tree.SearchAknn(p, e, knn.begin(), knn.end());
  //! // Optionally scale back to the actual metric distance.
  //! for (auto& nn : knn) { nn.second *= e; }
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchAknn(
      P const& x,
      Scalar const e,
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
      Index const k,
      Scalar const e,
      std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits::SpaceNpts(points_)));
    SearchAknn(x, e, knn->begin(), knn->end());
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dim)+k).
  //! \tparam P Point type.
  template <typename P>
  inline void SearchBox(
      P const& min, P const& max, std::vector<Index>* idxs) const {
    idxs->clear();
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there was overlap
    // then the search is slower. So unless many queries don't intersect there
    // is no point in adding it.

    SequenceBox root_box(root_box_);
    internal::SearchBox<Traits, Metric, Dim>(
        points_,
        metric_,
        indices_,
        Traits::PointCoords(min),
        Traits::PointCoords(max),
        idxs)(root_, &root_box);
  }

  //! \brief Point set used by the tree.
  inline Space const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static KdTree Load(Space points, std::string const& filename) {
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
  static KdTree Load(Space points, std::iostream* stream) {
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
    tree.Save(&s);
  }

 private:
  //! \brief Constructs a KdTree by reading its indexing and leaf information
  //! from a Stream.
  KdTree(Space points, internal::Stream* stream)
      : points_(std::move(points)),
        metric_(),
        nodes_(internal::MaxNodesFromPoints(Traits::SpaceNpts(points_))),
        indices_(Traits::SpaceNpts(points_)),
        root_(Load(stream)) {}

  inline void CalculateBoundingBox(SequenceBox* p_box) {
    auto& box = *p_box;
    auto sdim = Traits::SpaceSdim(points_);
    box.min.Fill(sdim, std::numeric_limits<Scalar>::max());
    box.max.Fill(sdim, std::numeric_limits<Scalar>::lowest());

    for (Index j = 0; j < Traits::SpaceNpts(points_); ++j) {
      Scalar const* const p = Traits::PointCoords(Traits::PointAt(points_, j));
      for (int i = 0; i < internal::Dimension<Traits, Dim>::Dim(points_); ++i) {
        Scalar const v = p[i];
        if (v < box.min[i]) {
          box.min[i] = v;
        }
        if (v > box.max[i]) {
          box.max[i] = v;
        }
      }
    }
  }

  //! \brief Builds a tree given a \p max_leaf_size and a Splitter.
  //! \details Run time may vary depending on the split strategy.
  inline Node* Build(Index const max_leaf_size) {
    assert(Traits::SpaceNpts(points_) > 0);
    assert(max_leaf_size > 0);

    std::iota(indices_.begin(), indices_.end(), 0);

    CalculateBoundingBox(&root_box_);

    Splitter splitter(points_, &indices_);
    return Builder{
        points_, indices_, max_leaf_size, splitter, &nodes_}(root_box_);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename P, typename V>
  inline void SearchNearest(
      Node const* const node, P const& x, V* visitor, EuclideanSpaceTag) const {
    internal::SearchNearestEuclidean<Traits, Metric, Dim, P, V>(
        points_, metric_, indices_, x, visitor)(node);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename P, typename V>
  inline void SearchNearest(
      Node const* const node,
      P const& x,
      V* visitor,
      TopologicalSpaceTag) const {
    internal::SearchNearestTopological<Traits, Metric, Dim, P, V>(
        points_, metric_, indices_, x, visitor)(node);
  }

  //! \brief Recursively reads the Node and its descendants.
  inline Node* ReadNode(internal::Stream* stream) {
    Node* node = nodes_.Allocate();
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
      Node const* const node, internal::Stream* stream) const {
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

  //! \private
  inline Node* Load(internal::Stream* stream) {
    stream->Read(&indices_);
    stream->Read(&root_box_.min.container());
    stream->Read(&root_box_.max.container());
    return ReadNode(stream);
  }

  //! \private
  inline void Save(internal::Stream* stream) const {
    stream->Write(indices_);
    stream->Write(root_box_.min.container());
    stream->Write(root_box_.max.container());
    WriteNode(root_, stream);
  }

  //! \brief Point set adapter used for querying point data.
  Space points_;
  //! \brief Metric used for comparing distances.
  Metric metric_;
  //! \brief Memory buffer for tree nodes.
  MemoryBuffer nodes_;
  //! \brief Sorted indices that refer to points inside points_.
  std::vector<Index> indices_;
  //! \brief Bounding box of the root node.
  SequenceBox root_box_;
  //! \brief Root of the KdTree.
  Node* root_;
};

}  // namespace pico_tree
