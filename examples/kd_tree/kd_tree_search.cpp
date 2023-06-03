#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/std_traits.hpp>

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
  //! \param d Point distance (that depends on the metric).
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
void Search3d() {
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
      tree.SearchNn(q, nn);
      tree.SearchKnn(q, 1, knn);
      tree.SearchRadius(q, search_radius_metric, knn, false);
      tree.SearchBox(min, max, idxs);
    }
  }

  Index k = 8;

  {
    ScopedTimer t("kd_tree aknn", run_count);
    for (Index i = 0; i < run_count; ++i) {
      // When the KdTree is created with the SlidingMidpointSplitter, ann
      // queries can be answered in O(1/e^d log n) time.
      tree.SearchAknn(q, k, max_error_ratio_metric, knn);
    }
  }

  {
    ScopedTimer t("kd_tree knn", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchKnn(q, k, knn);
    }
  }

  SearchNnCounter<pico_tree::Neighbor<Index, Scalar>> v(&nn);
  tree.SearchNearest(q, v);

  std::cout << "Custom visitor # nns considered: " << v.count() << std::endl;
}

// Search on the circle.
void SearchS1() {
  using PointX = Point1f;
  using TraitsX = pico_tree::StdTraits<std::vector<PointX>>;
  using NeighborX = pico_tree::KdTree<TraitsX, pico_tree::SO2>::NeighborType;

  const auto pi = typename PointX::ScalarType(3.1415926537);

  pico_tree::KdTree<TraitsX, pico_tree::SO2> tree(
      GenerateRandomN<PointX>(512, -pi, pi), 10);

  std::array<NeighborX, 8> knn;
  tree.SearchKnn(PointX{pi}, knn.begin(), knn.end());

  // These prints show that wrapping around near point -PI ~ PI is supported.
  std::cout << "Closest angles (index, distance, value): " << std::endl;
  for (auto const& nn : knn) {
    std::cout << "  " << nn.index << ", " << nn.distance << ", "
              << tree.points()[nn.index] << std::endl;
  }
}

int main() {
  Search3d();
  SearchS1();
  return 0;
}
