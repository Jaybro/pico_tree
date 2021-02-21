#pragma once

//#include "metric.hpp"
#include "kd_tree.hpp"

namespace pico_tree {

namespace internal {}

template <
    typename Index,
    typename Scalar,
    int Dim_,
    typename Points,
    typename Metric = MetricL2<Scalar, Dim_>>
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
    Index index;
    std::vector<Node*> children;
  };

  //! \brief This class contains (what we will call) the "doubling base" of
  //! the tree.
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

  //! \brief Creates a CoverTree given \p points and a doubling \p base.
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
  //! default MetricL2 results in squared distances.
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
  //! The amount of requested neighbors, k, should be sufficiently large to get
  //! a noticeable speed increase from this method. Within a leaf all points are
  //! compared to the query anyway, even if they are skipped. These calculations
  //! can be avoided by skipping leafs completely, which will never happen if
  //! all requested neighbors reside within a single one.
  //!
  //! Interpretation of both the input error ratio and output distances
  //! depend on the Metric. The default MetricL2 calculates squared
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

    Node* node = InsertFirstTwo();
    for (Index i = 2; i < points_.npts(); ++i) {
      node = Insert(node, i, &one, &other);
    }

    // std::cout << "one other " << one << " " << other << std::endl;
    // std::size_t size = 1;
    // NodeCount(node, &size);
    // std::cout << "tree size " << size << std::endl;

    // TODO Make cache friendly structure here. I.e., a depth first re-creation
    // of the tree.

    return node;
  }

  void NodeCount(Node const* const node, std::size_t* size) const {
    for (auto const& m : node->children) {
      NodeCount(m, size);
      (*size)++;
    }
  }

  //! \brief Inserts the first or first two nodes of the tree.
  //! \details Both papers don't really handle these case but here we go.
  inline Node* InsertFirstTwo() {
    Node* node = nodes_.Allocate();
    node->index = 0;
    if (points_.npts() == 1) {
      node->level = 0;
    } else {
      Scalar d = metric_(points_(0), points_(1));
      node->level = std::ceil(base_.Level(d));
      CreateChild(1, node);
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
      // intersecting the wrong balls.
      // TODO Change ParentDistance into MaxDistance(*m);
      // TODO The distance calculation can be cached. When SearchNeighbor is
      // called it's calculated again.
      if (visitor->max() >
          (metric_(p, points_(m->index)) - base_.ParentDistance(*m))) {
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
