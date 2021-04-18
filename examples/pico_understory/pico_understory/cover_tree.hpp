#pragma once

#include <cassert>
#include <numeric>
#include <pico_tree/internal/memory.hpp>
#include <pico_tree/internal/search_visitor.hpp>
#include <pico_tree/metric.hpp>
#include <random>

// Use this define to enable a simplified version of the nearest ancestor tree
// or disable it to use the regular one from "Faster Cover Trees".
//
// Testing seems to indicate that when the dataset has some structure (not
// random), the "simplified" version is faster.
// 1) Building: The performance difference can be large for a low leveling base,
// e.g., 1.3. The difference in building time becomes smaller when the leveling
// base increases.
// 2) Queries: Slower compared to the regular nearest ancestor tree for a low
// leveling base, but faster for a higher one.
//
// It seems that with structured data, both tree building and querying becomes
// faster when increasing the leveling base. Both times decreasing steadily when
// increasing the base. Here a value of 2.0 is the fastest.
//
// For random data, build and query times seem to be all over the place when
// steadily increasing the base. A value of 1.3 generally seems the fastest, but
// none of the values for this hyper parameter inspire trust.
#define SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE

namespace pico_tree {

template <typename Traits, typename Metric = L2<Traits>>
class CoverTree {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;

 public:
  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = Scalar;
  //! \brief CoverTree dimension. It equals pico_tree::kDynamicDim in case Dim
  //! is only known at run-time.
  static constexpr int Dim = Traits::Dim;
  //! \brief Traits with information about the input Spaces and Points.
  using TraitsType = Traits;
  //! \brief Point set or adaptor type.
  using SpaceType = Space;
  //! \brief The metric used for various searches.
  using MetricType = Metric;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<Index, Scalar>;

 private:
  struct Node {
    inline bool IsBranch() const { return !children.empty(); }
    inline bool IsLeaf() const { return children.empty(); }

    // TODO Could be moved to the tree.
    Scalar level;
    //! \brief Distance to the farthest child.
    Scalar max_distance;
    Index index;
    std::vector<Node*> children;
  };

  //! \brief This class contains (what we will call) the "leveling base" of the
  //! tree.
  //! \details It determines how fast the levels of the tree increase or
  //! decrease. When we raise "base" to the power of a certain natural number,
  //! that exponent represents the active level of the tree.
  //!
  //! The papers are written using a base of 2, but for performance reasons they
  //! use a base of 1.3.
  struct Base {
    inline Scalar CoverDistance(Node const& n) const {
      return std::pow(value, n.level);
    }

    //! Child distance is also the seperation distance.
    inline Scalar ChildDistance(Node const& n) const {
      return std::pow(value, n.level - Scalar(1.0));
    }

    inline Scalar ParentDistance(Node const& n) const {
      return std::pow(value, n.level + Scalar(1.0));
    }

    inline Scalar Level(Scalar const dst) const {
      return std::log(dst) / std::log(value);
    }

    Scalar value;
  };

 public:
  //! \brief The CoverTree cannot be copied.
  //! \details The CoverTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy. For now we are not interested in
  //! providing a deep copy.
  //! \private
  CoverTree(CoverTree const&) = delete;

  //! \brief Move constructor of the CoverTree.
  //! \details The move constructor is not implicitly created because of the
  //! deleted copy constructor.
  //! \private
  CoverTree(CoverTree&&) = default;

  //! \brief Creates a CoverTree given \p points and a leveling \p base.
  CoverTree(Space points, Scalar base)
      : points_(std::move(points)),
        metric_(),
        nodes_(Traits::SpaceNpts(points_)),
        base_{base},
        root_(Build()) {}

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<Index, Scalar>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2 metric results in Euclidean distances.
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
    SearchNearest(root_, x, &v);
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
  //! Metric. The default L2 results in squared distances.
  //! \tparam P Point type.
  //! \param x Input point.
  //! \param radius Search radius.
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
    SearchNearest(root_, x, &v);

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
  //! Interpretation of both the input error ratio and output distances depend
  //! on the Metric. The default L2 metric calculates Euclidean distances.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! Scalar max_error = Scalar(0.15);
  //! Scalar e = Scalar(1.0) + max_error;
  //! std::vector<Neighbor<Index, Scalar>> knn(k);
  //! tree.SearchAknn(x, e, knn.begin(), knn.end());
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
    SearchNearest(root_, x, &v);
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

  //! \brief Point set used by the tree.
  inline Space const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

 private:
  Node* Build() {
    assert(Traits::SpaceNpts(points_) > 0);

    // Both building and querying become a great deal faster for any of the
    // nearest ancestor cover trees when they are constructed with randomly
    // inserted points. It saves a huge deal of rebalancing (the build time of
    // the LiDAR dataset goes from 1.5 hour+ to about 3 minutes) and the tree
    // has a high probablity to be better balanced for queries.
    // For the simplified cover tree, query performance is greatly improved
    // using a randomized insertion at the price of construction time.
    Index const npts = Traits::SpaceNpts(points_);
    std::vector<Index> indices(npts);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Node* node = InsertFirstTwo(indices);
    for (Index i = 2; i < npts; ++i) {
      node = Insert(node, CreateNode(indices[i]));
    }

    // Cache friendly tree.
    // TODO Take 1. A better version is probably where the contents of the
    // vector are part of the node and not in a different memory blocks.
    internal::StaticBuffer<Node> nodes(npts);
    Node* root = DepthFirstBufferCopy(node, &nodes);
    std::swap(nodes_, nodes);
    node = root;

    // TODO This is quite expensive. We can do better by using the values
    // calculated during an insert.
    // Current version is well worth it vs. queries but maybe not for high
    // dimensions.
    UpdateMaxDistance(node);

    return node;
  }

  Node* DepthFirstBufferCopy(
      Node const* const node, internal::StaticBuffer<Node>* nodes) {
    Node* copy = nodes->Allocate();
    copy->index = node->index;
    copy->level = node->level;
    copy->children.reserve(node->children.size());

    for (auto const m : node->children) {
      copy->children.push_back(DepthFirstBufferCopy(m, nodes));
    }

    return copy;
  }

  void UpdateMaxDistance(Node* node) const {
    node->max_distance =
        MaxDistance(node, Traits::PointAt(points_, node->index));

    for (Node* m : node->children) {
      UpdateMaxDistance(m);
    }
  }

  template <typename P>
  Scalar MaxDistance(Node const* const node, P const& x) const {
    Scalar max = metric_(Traits::PointAt(points_, node->index), x);

    for (Node const* const m : node->children) {
      max = std::max(max, MaxDistance(m, x));
    }

    return max;
  }

  //! \brief Inserts the first or first two nodes of the tree.
  //! \details Both papers don't really handle these cases but here we go.
  inline Node* InsertFirstTwo(std::vector<Index> const& indices) {
    Node* node = nodes_.Allocate();
    node->index = indices[0];

    if (Traits::SpaceNpts(points_) == 1) {
      node->level = 0;
    } else {
      Scalar d = metric_(
          Traits::PointAt(points_, indices[0]),
          Traits::PointAt(points_, indices[1]));
      node->level = std::ceil(base_.Level(d));
      PushChild(node, CreateNode(indices[1]));
    }

    return node;
  }

  //! \brief Returns a new tree inserting \p node into \p tree.
  inline Node* Insert(Node* tree, Node* node) {
    auto const& x = Traits::PointAt(points_, node->index);
    Scalar d = metric_(Traits::PointAt(points_, tree->index), x);
    Scalar c = base_.CoverDistance(*tree);
    if (d > c) {
#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
      Scalar level = std::floor(base_.Level(c + d));

      while (tree->level < level) {
        tree = NodeToParent(RemoveLeaf(tree), tree);
      }
#else
      c *= base_.value;
      while (d > c) {
        tree = NodeToParent(RemoveLeaf(tree), tree);
        c = base_.ParentDistance(*tree);
        d = metric_(Traits::PointAt(points_, tree->index), x);
      }
#endif

      return NodeToParent(node, tree);
    } else {
      InsertCovered(tree, node);
      return tree;
    }
  }

  //! \brief Insert \p node somewhere in \p tree. If the new parent for \p
  //! node already has children, rebalancing may occur.
  inline void InsertCovered(Node* tree, Node* node) {
    // The following line may replace the contents of this function to get a
    // simplified cover tree:
    // PushChild(FindParent(tree, Traits::PointAt(points_, node->index)), node);

    auto const& x = Traits::PointAt(points_, node->index);
    Node* parent = FindParent(tree, x);

    if (parent->IsLeaf()) {
      PushChild(parent, node);
    } else {
      Rebalance(parent, node, x);
    }
  }

  template <typename P>
  inline Node* FindParent(Node* tree, P const& x) const {
    if (tree->IsLeaf()) {
      return tree;
    } else {
      // Traverse branches via the closest ancestors to the point. The paper
      // "Faster Cover Trees" mentions this as being part of the nearest
      // ancestor cover tree, but this speeds up the queries of the simplified
      // cover tree as well. Faster by ~70%, but tree creation is a bit slower
      // for it (3-14%).
      Scalar min_d =
          metric_(Traits::PointAt(points_, tree->children[0]->index), x);
      std::size_t min_i = 0;

      for (std::size_t i = 1; i < tree->children.size(); ++i) {
        Scalar d =
            metric_(Traits::PointAt(points_, tree->children[i]->index), x);

        if (d < min_d) {
          min_d = d;
          min_i = i;
        }
      }

      if (min_d <= base_.ChildDistance(*tree)) {
        return FindParent(tree->children[min_i], x);
      }

      return tree;
    }
  }

  template <typename P>
  void Rebalance(Node* parent, Node* node, P const& x) {
    std::vector<Node*> to_move;
    std::vector<Node*> to_stay;

    // For this node's children we want to know what the farthest distance of
    // their descendants is. That distance is this node's cover distance as it
    // is twice the separation distance. When the distance between the
    // children is more than twice this value, they don't intersect and
    // checking them can be skipped. However, since radius of this node's
    // sphere equals the cover distance, we always have to check all children.

    // TODO With the simplified nearest ancestor cover tree we can skip more
    // nodes, but perhaps we could use Node::max_distance later.

#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
    Scalar const d = base_.ChildDistance(*parent) * Scalar(2.0);

    for (auto child : parent->children) {
      auto const& y = Traits::PointAt(points_, child->index);

      if (metric_(x, y) < d) {
        Extract(x, y, child, &to_move, &to_stay);

        for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
          InsertCovered(child, *it);
        }

        to_stay.clear();
      }
    }

    node->level = parent->level - Scalar(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      InsertCovered(node, *it);
    }

    PushChild(parent, node);
#else
    for (auto& child : parent->children) {
      Extract(
          x, Traits::PointAt(points_, child->index), child, &to_move, &to_stay);

      for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
        child = Insert(child, *it);
      }

      to_stay.clear();
    }

    node->level = parent->level - Scalar(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      node = Insert(node, *it);
    }

    PushChild(parent, node);
#endif
  }

  //! \brief Fill the \p to_move and \p to_stay buffers based on if any \p
  //! descendant (of \p x_stay) is either closer to \p x_move or \p x_stay.
  //! Move is the newly inserted node for which we are rebalancing.
  template <typename P>
  void Extract(
      P const& x_move,
      P const& x_stay,
      Node* descendant,
      std::vector<Node*>* to_move,
      std::vector<Node*>* to_stay) {
    if (descendant->IsLeaf()) {
      return;
    }

    auto erase_begin = std::partition(
        descendant->children.begin(),
        descendant->children.end(),
        [this, &x_move, &x_stay](Node* node) -> bool {
          auto const& p = Traits::PointAt(points_, node->index);
          return metric_(x_stay, p) < metric_(x_move, p);
        });

    auto erase_rend =
        typename std::vector<Node*>::reverse_iterator(erase_begin);

    auto it = descendant->children.rbegin();
    for (; it != erase_rend; ++it) {
      to_move->push_back(*it);
      Strip(x_move, x_stay, *it, to_move, to_stay);
    }
    for (; it != descendant->children.rend(); ++it) {
      Extract(x_move, x_stay, *it, to_move, to_stay);
    }
    descendant->children.erase(erase_begin, descendant->children.end());
  }

  //! \brief Strips all descendants of d in a depth-first fashion and puts
  //! them in either the move or stay set.
  template <typename P>
  void Strip(
      P const& x_move,
      P const& x_stay,
      Node* descendant,
      std::vector<Node*>* to_move,
      std::vector<Node*>* to_stay) {
    if (descendant->IsLeaf()) {
      return;
    }

    for (auto const child : descendant->children) {
      Strip(x_move, x_stay, child, to_move, to_stay);

      auto const& p = Traits::PointAt(points_, child->index);
      if (metric_(x_stay, p) > metric_(x_move, p)) {
        to_move->push_back(child);
      } else {
        to_stay->push_back(child);
      }
    }
    descendant->children.clear();
  }

  inline Node* RemoveLeaf(Node* tree) const {
    assert(tree->IsBranch());

    Node* node;

    do {
      node = tree;
      tree = tree->children.back();
    } while (tree->IsBranch());

    node->children.pop_back();

    return tree;
  }

  inline Node* NodeToParent(Node* parent, Node* child) const {
    assert(parent->IsLeaf());

    parent->level = child->level + Scalar(1.0);
    parent->children.push_back(child);
    return parent;
  }

  inline Node* CreateNode(Index idx) {
    Node* node = nodes_.Allocate();
    node->index = idx;
    return node;
  }

  inline void PushChild(Node* parent, Node* child) {
    child->level = parent->level - Scalar(1.0);
    parent->children.push_back(child);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNearest(
      Node const* const node, P const& x, V* visitor) const {
    Scalar const d = metric_(x, Traits::PointAt(points_, node->index));
    if (visitor->max() > d) {
      (*visitor)(node->index, d);
    }

    std::vector<std::pair<Node const*, Scalar>> sorted;
    sorted.reserve(node->children.size());
    for (auto const child : node->children) {
      sorted.push_back(
          {child, metric_(x, Traits::PointAt(points_, child->index))});
    }

    std::sort(
        sorted.begin(),
        sorted.end(),
        [](std::pair<Node const*, Scalar> const& a,
           std::pair<Node const*, Scalar> const& b) -> bool {
          return a.second < b.second;
        });

    for (auto const& m : sorted) {
      // Algorithm 1 from paper "Faster Cover Trees" has a mistake. It checks
      // with respect to the nearest point, not the query point itself,
      // intersecting the wrong spheres.
      // Algorithm 1 from paper "Cover Trees for Nearest Neighbor" is correct.

      // The upper-bound distance a descendant can be is twice the cover
      // distance of the node. This is true taking the invariants into account.
      // NOTE: In "Cover Trees for Nearest Neighbor" this upper-bound
      // practically appears to be half this distance, as new nodes are only
      // added when they are within cover distance.
      // For "Faster Cover Trees" it is twice the cover distance due to the
      // first phase of the insert algorithm (not having a root at infinity).

      // TODO The distance calculation can be cached. When SearchNeighbor is
      // called it's calculated again.
      if (visitor->max() >
          (metric_(x, Traits::PointAt(points_, m.first->index)) -
           m.first->max_distance)) {
        SearchNearest(m.first, x, visitor);
      }
    }
  }

  //! Point set adapter used for querying point data.
  Space points_;
  //! Metric used for comparing distances.
  Metric metric_;
  //! Memory buffer for tree nodes.
  internal::StaticBuffer<Node> nodes_;
  //! Base
  Base base_;
  //! Root of the CoverTree.
  Node const* const root_;
};

}  // namespace pico_tree
