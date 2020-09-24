#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

template <typename PicoAdaptor>
using KdTreeCt = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using KdTreeRt = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    pico_tree::kRuntimeDims,
    PicoAdaptor>;

// Compile time or run time known dimensions.
void Build() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoAdaptorX adaptor(random);

  {
    ScopedTimer t("build kd_tree ct");
    KdTreeCt<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }

  {
    ScopedTimer t("build kd_tree rt");
    KdTreeRt<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

//! \brief Search visitor for finding an approximate nearest neighbor.
//! \details This visitor updates an approximate nearest neighbor that will at
//! most be a distance ratio away from the true nearest neighbor once the
//! search has finished.
template <typename Index, typename Scalar>
class SearchAnn {
 public:
  //! Creates a visitor for approximate nearest neighbor searching.
  //! \param e Maximum distance error ratio.
  //! \param ann Search result.
  SearchAnn(Scalar const e, std::pair<Index, Scalar>* ann)
      : e_{Scalar(1.0) / e}, ann_{*ann} {
    // Initial search distance.
    ann_ = {0, std::numeric_limits<Scalar>::max()};
  }

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) const {
    // The distance is scaled to be d = d / e.
    ann_ = std::make_pair(idx, d * e_);
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return ann_.second; }

 private:
  Scalar const e_;
  std::pair<Index, Scalar>& ann_;
};

// Different search options.
void Searches() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index run_count = 1024 * 1024;
  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoAdaptorX adaptor(random);
  KdTreeCt<PicoAdaptorX> tree(adaptor, max_leaf_size);

  Scalar min_v = 25.1f;
  Scalar max_v = 37.9f;
  PointX min, max, pnn;
  min.Fill(min_v);
  max.Fill(max_v);
  pnn.Fill((max_v + min_v) / 2.0f);

  Index k = 4;
  Scalar search_radius = 2.0;
  // When building KdTree with default template arguments, this is the squared
  // distance.
  Scalar search_radius_metric = tree.metric()(search_radius);
  // The ann can not be further away than a factor of (1 + max_error_percentage)
  // from the real nn.
  Scalar max_error_percentage = 0.2;
  // Apply the metric to the max ratio difference.
  Scalar max_error_ratio_metric =
      tree.metric()(Scalar(1.0) + max_error_percentage);

  std::vector<std::pair<Index, Scalar>> nn;
  std::vector<Index> idxs;
  std::pair<Index, Scalar> ann;

  ScopedTimer t("kd_tree", run_count);
  for (Index i = 0; i < run_count; ++i) {
    tree.SearchKnn(pnn, k, &nn, false);
    tree.SearchRadius(pnn, search_radius_metric, &nn, false);
    tree.SearchBox(min, max, &idxs);

    // When the KdTree is created with the SlidingMidpointSplitter, ann queries
    // can be answered in O(1/e^d log n) time.
    SearchAnn<Index, Scalar> ann_visitor(max_error_ratio_metric, &ann);
    tree.SearchNn(pnn, &ann_visitor);
    // The actual distance according to tree.metric().
    ann.second *= max_error_ratio_metric;
  }
}

int main() {
  Build();
  Searches();
  return 0;
}
