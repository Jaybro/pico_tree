#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Search visitor that counts how many points were considered as a possible
// nearest neighbor.
template <typename Neighbor_>
class search_nn_counter {
 public:
  using neighbor_type = Neighbor_;
  using index_type = typename Neighbor_::index_type;
  using scalar_type = typename Neighbor_::scalar_type;

  // Create a visitor for approximate nearest neighbor searching. The argument
  // is the search result.
  inline search_nn_counter(neighbor_type& nn) : count_(0), nn_(nn) {
    // Initial search distance.
    nn_.distance = std::numeric_limits<scalar_type>::max();
  }

  // Visit current point. This method is required. The search algorithm calls
  // this function for every point it encounters in the kd_tree. The arguments
  // of the method are respectively the index and distance of the visited point.
  inline void operator()(index_type const idx, scalar_type const dst) {
    // Only update the nearest neighbor when the point we visit is actually
    // closer to the query point.
    if (max() > dst) {
      nn_ = {idx, dst};
    }
    count_++;
  }

  // Maximum search distance with respect to the query point. This method is
  // required. The nodes of the kd_tree are filtered using this method.
  inline scalar_type const& max() const { return nn_.distance; }

  // The amount of points visited during a query.
  inline index_type const& count() const { return count_; }

 private:
  index_type count_;
  neighbor_type& nn_;
};

int main() {
  using point = pico_tree::point_2f;
  using scalar = typename point::scalar_type;

  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  scalar area_size = 1000;

  using kd_tree = pico_tree::kd_tree<std::vector<point>>;
  using neighbor = typename kd_tree::neighbor_type;

  kd_tree tree(
      pico_tree::generate_random_n<point>(point_count, area_size),
      max_leaf_size);

  point q{area_size / scalar(2.0), area_size / scalar(2.0)};
  neighbor nn;
  search_nn_counter<neighbor> v(nn);
  tree.search_nearest(q, v);

  std::cout << "Number of points visited: " << v.count() << std::endl;

  return 0;
}
