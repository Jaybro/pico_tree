#pragma once

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

    std::size_t one = 0, other = 0;

    // For the simplified cover tree, query performance is greatly improved
    // using a randomized insertion. The tree seems to get highly imbalanced if
    // not done. Building the tree becomes a lot slower but well worth it.
    // TODO May be solved by the nearest ancestor version.
    std::vector<Index> indices(points_.npts());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Node* node = InsertFirstTwo(indices);
    for (Index i = 2; i < points_.npts(); ++i) {
      node = Insert(node, indices[i], &one, &other);
    }

    // std::cout << "one other " << one << " " << other << std::endl;

    // TODO Make cache friendly structure here. I.e., a depth first re-creation
    // of the tree.

    // TODO This is quite expensive. Perhaps we can do better. Well worth it vs.
    // queries. These are otherwise extremely slow. ~70% faster.
    UpdateMaxDistance(node);

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
      CreateChild(indices[1], node);
    }

    return node;
  }

  //! \brief Inserts a new node for point \p idx into the tree defined by \p n.
  inline Node* Insert(
      Node* n, Index const idx, std::size_t* one, std::size_t* other) {
    // TODO Change to levels?
    Scalar d = metric_(points_(n->index), points_(idx));
    Scalar c = base_.CoverDistance(*n);
    if (d > c) {
      c *= base_.value;
      while (d > c) {
        n = ChildToParent(RemoveLeaf(n), n);
        c = base_.ParentDistance(*n);
      }

      *one += 1;

      return CreateParent(idx, n);
    } else {
      *other += 1;

      InsertLeaf(n, idx);
      return n;
    }
  }

  inline void InsertLeaf(Node* n, Index const idx) {
    CreateChild(idx, FindParent(n, points_(idx)));
  }

  template <typename P>
  inline Node* FindParent(Node* n, P const& p) const {
    Scalar s = base_.ChildDistance(*n);
    for (std::size_t i = 0; i < n->children.size(); ++i) {
      Node* m = n->children[i];
      if (metric_(points_(m->index), p) <= s) {
        return FindParent(m, p);
      }
    }

    return n;
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

  inline Node* ChildToParent(Node* p, Node* c) const {
    assert(p->IsLeaf());

    p->level = c->level + Scalar(1.0);
    p->children.push_back(c);
    return p;
  }

  inline Node* CreateParent(Index idx, Node* c) {
    Node* p = nodes_.Allocate();
    p->index = idx;
    return ChildToParent(p, c);
  }

  inline void CreateChild(Index idx, Node* p) {
    Node* c = nodes_.Allocate();
    c->index = idx;
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
    // query point.
    for (auto const& m : node->children) {
      // Algorithm 1 from paper "Faster Cover Trees" has a mistake. It checks
      // with respect to the nearest point, not the query point itself,
      // intersecting the wrong spheres.
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
