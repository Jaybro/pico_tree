#pragma once

#include <numeric>
#include <random>

#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

namespace internal {}

template <
    typename Index,
    typename Scalar,
    int Dim_,
    typename Points,
    typename Metric = L2<Scalar, Dim_>>
class CoverTree {
 public:
  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = Scalar;
  //! \brief CoverTree dimension. It equals pico_tree::kDynamicDim in case Dim
  //! is only known at run-time.
  static constexpr int Dim = Dim_;
  //! \brief Point set or adaptor type.
  using PointsType = Points;
  //! \brief The metric used for various searches.
  using MetricType = Metric;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<Index, Scalar>;

 private:
  struct Node {
    inline bool IsBranch() const { return !children.empty(); }
    inline bool IsLeaf() const { return children.empty(); }

    // TODO Maybe move to the tree itself.
    // TODO Could possibly reduce storage if changed to some integer type at the
    // price of performance?
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
  CoverTree(Points points, Scalar base)
      : points_(std::move(points)),
        metric_(points_.sdim()),
        nodes_(points_.npts()),
        base_{base},
        root_(Build()) {}

  //! \brief Searches for the k nearest neighbors of point \p p , where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<Index, Scalar>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2 metric results in Euclidean distances.
  //! \tparam P Point type.
  //! \tparam RandomAccessIterator Iterator type.
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& p, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchKnn<RandomAccessIterator> v(begin, end);
    SearchNearest(root_, p, &v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p p and stores
  //! the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& p, Index const k, std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, points_.npts()));
    SearchKnn(p, knn->begin(), knn->end());
  }

  //! \brief Searches for all the neighbors of point \p p that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! Metric. The default L2 results in squared distances.
  //! \tparam P Point type.
  //! \param p Input point.
  //! \param radius Search radius.
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& p,
      Scalar const radius,
      std::vector<NeighborType>* n,
      bool const sort = false) const {
    internal::SearchRadius<NeighborType> v(radius, n);
    SearchNearest(root_, p, &v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p p ,
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
  //! tree.SearchAknn(p, e, knn.begin(), knn.end());
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchAknn(
      P const& p,
      Scalar const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNearest(root_, p, &v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p p
  //! and stores the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchAknn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchAknn(
      P const& p,
      Index const k,
      Scalar const e,
      std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, points_.npts()));
    SearchAknn(p, e, knn->begin(), knn->end());
  }

  //! \brief Point set used by the tree.
  inline Points const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

 private:
  Node* Build() {
    assert(points_.npts() > 0);

    // For the simplified cover tree, query performance is greatly improved
    // using a randomized insertion at the price of construction time. The tree
    // seems to get highly imbalanced if not done.
    // TODO May be solved by the nearest ancestor version.
    std::vector<Index> indices(points_.npts());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Node* node = InsertFirstTwo(indices);
    for (Index i = 2; i < points_.npts(); ++i) {
      node = Insert(node, CreateNode(indices[i]));
    }

    // TODO This is quite expensive. Perhaps we can do better. Well worth it vs.
    // queries. These are otherwise extremely slow. ~70% faster.
    UpdateMaxDistance(node);

    // TODO Make cache friendly structure here. I.e., a depth first re-creation
    // of the tree.

    return node;
  }

  void UpdateMaxDistance(Node* node) const {
    node->max_distance = MaxDistance(node, points_(node->index));

    for (Node* m : node->children) {
      UpdateMaxDistance(m);
    }
  }

  template <typename P>
  Scalar MaxDistance(Node const* const node, P const& p) const {
    Scalar max = metric_(points_(node->index), p);

    for (Node const* const m : node->children) {
      max = std::max(max, MaxDistance(m, p));
    }

    return max;
  }

  //! \brief Inserts the first or first two nodes of the tree.
  //! \details Both papers don't really handle these cases but here we go.
  inline Node* InsertFirstTwo(std::vector<Index> const& indices) {
    Node* node = nodes_.Allocate();
    node->index = indices[0];
    if (points_.npts() == 1) {
      node->level = 0;
    } else {
      Scalar d = metric_(points_(indices[0]), points_(indices[1]));
      node->level = std::ceil(base_.Level(d));
      PushChild(node, CreateNode(indices[1]));
    }

    return node;
  }

  //! \brief Inserts a new node for point \p idx into the tree defined by \p n.
  inline Node* Insert(Node* n, Node* x) {
    auto const& p = points_(x->index);
    Scalar d = metric_(points_(n->index), p);
    Scalar c = base_.CoverDistance(*n);
    if (d > c) {
      c *= base_.value;
      while (d > c) {
        n = NodeToParent(RemoveLeaf(n), n);
        c = base_.ParentDistance(*n);
        d = metric_(points_(n->index), p);
      }

      return NodeToParent(x, n);
    } else {
      InsertLeaf(n, x);
      return n;
    }
  }

  inline void InsertLeaf(Node* n, Node* l) {
    // Change contents of this function to the following line for a simplified
    // cover tree.
    // PushChild(FindParent(n, points_(l->index)), l);

    auto const& x = points_(l->index);
    Node* p = FindParent(n, x);

    if (p->IsLeaf()) {
      PushChild(p, l);
    } else {
      Rebalance(p, l, x);
    }
  }

  template <typename P>
  inline Node* FindParent(Node* n, P const& p) const {
    if (n->IsLeaf()) {
      return n;
    } else {
      // Traverse branches via the closest ancestors to the point. The paper
      // "Faster Cover Trees" mentions this as being part of the nearest
      // ancestor cover tree, but this speeds up the queries of the simplified
      // cover tree as well. Faster by ~70%, but tree creation is a bit slower
      // for it (3-14%).
      Scalar min_d = metric_(points_(n->children[0]->index), p);
      Index min_i = 0;

      for (Index i = 1; i < n->children.size(); ++i) {
        Scalar d = metric_(points_(n->children[i]->index), p);

        if (d < min_d) {
          min_d = d;
          min_i = i;
        }
      }

      if (min_d <= base_.ChildDistance(*n)) {
        return FindParent(n->children[min_i], p);
      }

      return n;
    }
  }

  template <typename P>
  void Rebalance(Node* n, Node* c, P const& p) {
    std::vector<Node*> to_move;
    std::vector<Node*> to_stay;

    // For this node's children we want to know what the farthest distance of
    // their descendants is. That distance is this node's cover distance as it
    // is twice the separation distance. When the distance between the children
    // is more than twice this value, they don't intersect and checking them can
    // be skipped. However, since radius of this node's sphere equals the cover
    // distance, we always have to check all children.
    // TODO The only reason the above is true, is because Insert does not always
    // return a node that contains all its descendants in its cover distance. If
    // it's possible to change Insert to always return a node that covers all
    // its descendants with its cover distance, this algorithm could be faster.
    // 1) Can actually skip children, especially at higher dimensions.
    // 2) We'll never have to level the tree from Rebalance and can use
    // InsertLeaf everywhere instead of Insert.

    for (auto& m : n->children) {
      Extract(points_(m->index), p, m, &to_move, &to_stay);

      for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
        m = Insert(m, *it);
      }

      to_stay.clear();
    }

    c->level = n->level - Scalar(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      c = Insert(c, *it);
    }

    PushChild(n, c);
  }

  template <typename P>
  void Extract(
      P const& n,
      P const& p,
      Node* d,
      std::vector<Node*>* to_move,
      std::vector<Node*>* to_stay) {
    if (d->IsLeaf()) {
      return;
    }
    auto erase_begin = std::partition(
        d->children.begin(),
        d->children.end(),
        [this, &n, &p](Node* r) -> bool {
          return metric_(n, points_(r->index)) < metric_(p, points_(r->index));
        });

    auto erase_rend =
        typename std::vector<Node*>::reverse_iterator(erase_begin);

    auto it = d->children.rbegin();
    for (; it != erase_rend; ++it) {
      to_move->push_back(*it);
      Strip(n, p, *it, to_move, to_stay);
    }
    for (; it != d->children.rend(); ++it) {
      Extract(n, p, *it, to_move, to_stay);
    }
    d->children.erase(erase_begin, d->children.end());
  }

  //! \brief Strips all descendants of d in a depth-first fashion and puts them
  //! in either the move or stay set.
  template <typename P>
  void Strip(
      P const& n,
      P const& p,
      Node* d,
      std::vector<Node*>* to_move,
      std::vector<Node*>* to_stay) {
    if (d->IsLeaf()) {
      return;
    }

    for (auto const& r : d->children) {
      Strip(n, p, r, to_move, to_stay);
      if (metric_(n, points_(r->index)) > metric_(p, points_(r->index))) {
        to_move->push_back(r);
      } else {
        to_stay->push_back(r);
      }
    }
    d->children.clear();
  }

  // Called with trust.
  inline Node* RemoveLeaf(Node* n) const {
    assert(n->IsBranch());

    Node* m;

    do {
      m = n;
      n = n->children.back();
    } while (n->IsBranch());

    m->children.pop_back();

    return n;
  }

  inline Node* NodeToParent(Node* p, Node* c) const {
    assert(p->IsLeaf());

    p->level = c->level + Scalar(1.0);
    p->children.push_back(c);
    return p;
  }

  inline Node* CreateNode(Index idx) {
    Node* c = nodes_.Allocate();
    c->index = idx;
    return c;
  }

  inline void PushChild(Node* p, Node* c) {
    c->level = p->level - Scalar(1.0);
    p->children.push_back(c);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p p depending
  //! on their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNearest(
      Node const* const node, P const& p, V* visitor) const {
    Scalar const d = metric_(p, points_(node->index));
    if (visitor->max() > d) {
      (*visitor)(node->index, d);
    }

    // TODO The paper tells us to sort the children based on distance from the
    // query point. We do the same in the KdTree already. Here it's more of the
    // same.
    for (auto const& m : node->children) {
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
      if (visitor->max() > (metric_(p, points_(m->index)) - m->max_distance)) {
        SearchNearest(m, p, visitor);
      }
    }
  }

  //! Point set adapter used for querying point data.
  Points points_;
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
