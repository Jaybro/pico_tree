#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>

// PicoTree provides default support for std::vector<PointType> as long as a
// traits class for PointType is implemented. See include
// <pico_toolshed/point.hpp> for an example of such traits.
//
// The KdTree can either fully own the vector or it can be taken by reference.
void BasicVector() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;

  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;

  {
    ScopedTimer t("build kd_tree val");

    // This version of pico_tree::StdTraits can be used to either move or copy a
    // vector of points into the tree. In this example it's moved.
    pico_tree::KdTree<pico_tree::StdTraits<std::vector<PointX>>> tree(
        GenerateRandomN<PointX>(point_count, area_size), max_leaf_size);

    pico_tree::Neighbor<Index, Scalar> nn;
    tree.SearchNn(tree.points()[0], &nn);
  }

  {
    ScopedTimer t("build kd_tree ref");
    auto random = GenerateRandomN<PointX>(point_count, area_size);

    // To prevent a copy, use the pico_tree::StdTraits with an
    // std::reference_wrapper.
    pico_tree::KdTree<
        pico_tree::StdTraits<std::reference_wrapper<std::vector<PointX>>>>
        tree(random, max_leaf_size);

    pico_tree::Neighbor<Index, Scalar> nn;
    tree.SearchNn(random[0], &nn);
  }
}

//! \brief Search visitor that counts how many points were considered as a
//! nearest neighbor.
template <typename Neighbor>
class SearchNnCounter {
 private:
  using Index = typename Neighbor::IndexType;
  using Scalar = typename Neighbor::ScalarType;

 public:
  //! \brief Creates a visitor for approximate nearest neighbor searching.
  //! \param nn Search result.
  inline SearchNnCounter(Neighbor* nn) : count_(0), nn_(*nn) {
    // Initial search distance.
    nn_.distance = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  //! \details This method is required. The KdTree calls this function when it
  //! finds a point that is closer to the query than the result of this
  //! visitors' max() function. I.e., it found a new nearest neighbor.
  //! \param idx Point index.
  //! \param d Point distance
  //! (that depends on the metric).
  inline void operator()(Index const idx, Scalar const dst) {
    count_++;
    nn_ = {idx, dst};
  }

  //! \brief Maximum search distance with respect to the query point.
  //! \details This method is required.
  inline Scalar const& max() const { return nn_.distance; }

  //! \brief Returns the amount of points that were considered the nearest
  //! neighbor.
  //! \details This method is not required.
  inline Index const& count() const { return count_; }

 private:
  Index count_;
  Neighbor& nn_;
};

// Different search options.
void Search() {
  using PointX = Point3f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;

  Index run_count = 1024 * 1024;
  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;

  pico_tree::KdTree<pico_tree::StdTraits<std::vector<PointX>>> tree(
      GenerateRandomN<PointX>(point_count, area_size), max_leaf_size);

  Scalar min_v = 25.1f;
  Scalar max_v = 37.9f;
  PointX min, max, q;
  min.Fill(min_v);
  max.Fill(max_v);
  q.Fill((max_v + min_v) / 2.0f);

  Scalar search_radius = 2.0f;
  // When building KdTree with default template arguments, this is the squared
  // distance.
  Scalar search_radius_metric = tree.metric()(search_radius);
  // The ann can not be further away than a factor of (1 + max_error_percentage)
  // from the real nn.
  Scalar max_error_percentage = 0.2f;
  // Apply the metric to the max ratio difference.
  Scalar max_error_ratio_metric = tree.metric()(1.0f + max_error_percentage);

  pico_tree::Neighbor<Index, Scalar> nn;
  std::vector<pico_tree::Neighbor<Index, Scalar>> knn;
  std::vector<Index> idxs;

  {
    ScopedTimer t("kd_tree nn, radius and box", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchNn(q, &nn);
      tree.SearchKnn(q, 1, &knn);
      tree.SearchRadius(q, search_radius_metric, &knn, false);
      tree.SearchBox(min, max, &idxs);
    }
  }

  Index k = 8;

  {
    ScopedTimer t("kd_tree aknn", run_count);
    for (Index i = 0; i < run_count; ++i) {
      // When the KdTree is created with the SlidingMidpointSplitter, ann
      // queries can be answered in O(1/e^d log n) time.
      tree.SearchAknn(q, k, max_error_ratio_metric, &knn);
    }
  }

  {
    ScopedTimer t("kd_tree knn", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchKnn(q, k, &knn);
    }
  }

  SearchNnCounter<pico_tree::Neighbor<Index, Scalar>> v(&nn);
  tree.SearchNearest(q, &v);

  std::cout << "Custom visitor # nns considered: " << v.count() << std::endl;
}

int main() {
  BasicVector();
  Search();
  return 0;
}
