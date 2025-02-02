#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Search visitor that counts how many points were considered as a possible
// nearest neighbor.
template <typename Neighbor>
class SearchNnCounter {
 public:
  using NeighborType = Neighbor;
  using IndexType = typename Neighbor::IndexType;
  using ScalarType = typename Neighbor::ScalarType;

  // Create a visitor for approximate nearest neighbor searching. The argument
  // is the search result.
  inline SearchNnCounter(Neighbor& nn) : count_(0), nn_(nn) {
    // Initial search distance.
    nn_.distance = std::numeric_limits<ScalarType>::max();
  }

  // Visit current point. This method is required. The search algorithm calls
  // this function for every point it encounters in the KdTree. The arguments of
  // the method are respectively the index and distance of the visited point.
  inline void operator()(IndexType const idx, ScalarType const dst) {
    // Only update the nearest neighbor when the point we visit is actually
    // closer to the query point.
    if (max() > dst) {
      nn_ = {idx, dst};
    }
    count_++;
  }

  // Maximum search distance with respect to the query point. This method is
  // required. The nodes of the KdTree are filtered using this method.
  inline ScalarType const& max() const { return nn_.distance; }

  // The amount of points visited during a query.
  inline IndexType const& count() const { return count_; }

 private:
  IndexType count_;
  Neighbor& nn_;
};

int main() {
  using PointX = Point2f;
  using Scalar = typename PointX::ScalarType;

  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  Scalar area_size = 1000;

  using KdTree = pico_tree::KdTree<std::vector<PointX>>;
  using Neighbor = typename KdTree::NeighborType;

  KdTree tree(GenerateRandomN<PointX>(point_count, area_size), max_leaf_size);

  PointX q{area_size / Scalar(2.0), area_size / Scalar(2.0)};
  Neighbor nn;
  SearchNnCounter<Neighbor> v(nn);
  tree.SearchNearest(q, v);

  std::cout << "Number of points visited: " << v.count() << std::endl;

  return 0;
}
