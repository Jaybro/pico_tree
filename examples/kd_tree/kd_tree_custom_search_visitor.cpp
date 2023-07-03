#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

//! \brief Search visitor that counts how many points were considered as a
//! nearest neighbor.
template <typename Neighbor>
class SearchNnCounter {
 public:
  using NeighborType = Neighbor;
  using IndexType = typename Neighbor::IndexType;
  using ScalarType = typename Neighbor::ScalarType;

  //! \brief Creates a visitor for approximate nearest neighbor searching.
  //! \param nn Search result.
  inline SearchNnCounter(Neighbor& nn) : count_(0), nn_(nn) {
    // Initial search distance.
    nn_.distance = std::numeric_limits<ScalarType>::max();
  }

  //! \brief Visit current point.
  //! \details This method is required. The KdTree calls this function when it
  //! finds a point that is closer to the query than the result of this
  //! visitors' max() function. I.e., it found a new nearest neighbor.
  //! \param idx Point index.
  //! \param d Point distance (that depends on the metric).
  inline void operator()(IndexType const idx, ScalarType const dst) {
    count_++;
    nn_ = {idx, dst};
  }

  //! \brief Maximum search distance with respect to the query point.
  //! \details This method is required.
  inline ScalarType const& max() const { return nn_.distance; }

  //! \brief Returns the number of points that were considered the nearest
  //! neighbor.
  //! \details This method is not required.
  inline IndexType const& count() const { return count_; }

 private:
  IndexType count_;
  Neighbor& nn_;
};

int main() {
  using PointX = Point2f;
  using Scalar = typename PointX::ScalarType;

  std::size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  Scalar area_size = 1000;

  using KdTree = pico_tree::KdTree<std::vector<PointX>>;
  using Neighbor = typename KdTree::NeighborType;

  KdTree tree(GenerateRandomN<PointX>(point_count, area_size), max_leaf_size);

  PointX q{area_size / Scalar(2.0), area_size / Scalar(2.0)};
  Neighbor nn;
  SearchNnCounter<Neighbor> v(nn);
  tree.SearchNearest(q, v);

  std::cout << "Custom visitor # nns considered: " << v.count() << std::endl;

  return 0;
}
