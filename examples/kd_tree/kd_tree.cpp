#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

// Compile time or run time known dimensions.
void Build() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  // Because the adapter only contains a reference to the vector, we don't care
  // that it gets copied twice by both trees below.
  PicoAdaptorX adaptor(random);

  {
    ScopedTimer t("build kd_tree ct");
    constexpr int Dim = PointX::Dim;
    pico_tree::KdTree<Index, Scalar, Dim, PicoAdaptorX> tree(
        adaptor, max_leaf_size);
  }

  {
    ScopedTimer t("build kd_tree rt");
    constexpr int Dim = pico_tree::kDynamicDim;
    pico_tree::KdTree<Index, Scalar, Dim, PicoAdaptorX> tree(
        adaptor, max_leaf_size);
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
  constexpr int Dim = PointX::Dim;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index run_count = 1024 * 1024;
  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  // The tree can fully own the adaptor!
  pico_tree::KdTree<Index, Scalar, Dim, PicoAdaptorX> tree(
      PicoAdaptorX(random), max_leaf_size);

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
  Build();
  Search();
  return 0;
}
