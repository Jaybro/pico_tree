#include <opencv2/flann.hpp>

#include "benchmark.hpp"

// BmOpenCvFlann benchmarks the OpenCV version of FLANN. It is possible to
// simply include the original FLANN <flann/flann.hpp> and replace the cvflann
// namespace by the flann namespace and the benchmark will still work.
// NOTE: The OpenCV version of FLANN performs quite a bit faster for queries
// than https://github.com/flann-lib/flann version 1.9.1. About an order of
// magnitude for a single NN. Tree build times are the same.
class BmOpenCvFlann : public pico_tree::Benchmark {
 public:
};

// Change the namespace below to switch between FLANN versions.
namespace fl = cvflann;
// namespace fl = flann;

// The OpenCV version of FLANN has improved query performance over the original
// but does not include the flann::L2_3D<scalar_type> distance class:
// https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/dist.h
// This distance class gives a reasonable performance boost over
// (cv)flann::L2_Simple because it uses a compile time constant Dim_ count.
// NOTE: Strictly speaking it shouldn't be part of the performance test.
namespace cvflann {

template <class T>
struct L2_3D {
  typedef bool is_kdtree_distance;

  typedef T ElementType;
  typedef typename Accumulator<T>::Type ResultType;

  template <typename Iterator1_, typename Iterator2_>
  ResultType operator()(
      Iterator1_ a,
      Iterator2_ b,
      [[maybe_unused]] size_t size,
      ResultType /*worst_dist*/ = -1) const {
    ResultType result = ResultType();
    ResultType diff;
    diff = *a++ - *b++;
    result += diff * diff;
    diff = *a++ - *b++;
    result += diff * diff;
    diff = *a++ - *b++;
    result += diff * diff;
    return result;
  }

  template <typename U, typename V>
  inline ResultType accum_dist(const U& a, const V& b, int) const {
    return (a - b) * (a - b);
  }
};

}  // namespace cvflann

template <typename Scalar_, std::size_t Dim_>
Scalar_* raw_copy(std::vector<pico_tree::point<Scalar_, Dim_>> const& points) {
  Scalar_* copy = new Scalar_[points.size() * Dim_];
  std::copy(
      points.begin()->data(),
      points.begin()->data() + points.size() * Dim_,
      copy);
  return copy;
}

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmOpenCvFlann, BuildRt)(benchmark::State& state) {
  int max_leaf_size = static_cast<int>(state.range(0));

  // Reorder will change the order of the input to fit the generated indices,
  // but it will replace (delete) the original input. It is set to true because
  // we use it to improve query times during the Knn test.
  // The tree takes ownership of the copied data.
  // NOTE: One could argue the copy is part of the benchmark, but didn't add it.
  fl::Matrix<scalar_type> matrix(
      raw_copy(points_tree_), points_tree_.size(), 3);
  fl::KDTreeSingleIndexParams pindex(max_leaf_size, true);

  for (auto _ : state) {
    fl::KDTreeSingleIndex<fl::L2_3D<scalar_type>> tree(matrix, pindex);
    tree.buildIndex();
  }
}

BENCHMARK_REGISTER_F(BmOpenCvFlann, BuildRt)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(BmOpenCvFlann, KnnCt)(benchmark::State& state) {
  int max_leaf_size = static_cast<int>(state.range(0));
  std::size_t knn_count = static_cast<std::size_t>(state.range(1));

  // Reorder will change the order of the input to fit the generated indices,
  // but it will replace (and delete) the original input. The reorder option
  // does has a small positive effect on the performance of the queries.
  // The tree takes ownership of the copied data.
  fl::Matrix<scalar_type> matrix(
      raw_copy(points_tree_), points_tree_.size(), 3);
  fl::KDTreeSingleIndexParams pindex(max_leaf_size, true);
  // "Custom" L2_3D metric has a decent positive effect on query times.
  fl::KDTreeSingleIndex<fl::L2_3D<scalar_type>> tree(matrix, pindex);
  tree.buildIndex();

  // Search all nodes, no approximate search and no sorting.
  fl::SearchParams psearch(-1, 0.0f, false);

  // There is also the option to query them all at once, but this doesn't really
  // change performance and this version looks more like the other benchmarks.
  for (auto _ : state) {
    // The only supported index type is int.
    std::vector<int> indices(knn_count);
    std::vector<scalar_type> distances(knn_count);
    fl::Matrix<int> mat_indices(indices.data(), 1, knn_count);
    fl::Matrix<scalar_type> mat_distances(distances.data(), 1, knn_count);

    for (auto& p : points_test_) {
      fl::Matrix<scalar_type> query(p.data(), 1, 3);
      tree.knnSearch(
          query,
          mat_indices,
          mat_distances,
          static_cast<int>(knn_count),
          psearch);
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: K nearest neighbors.
BENCHMARK_REGISTER_F(BmOpenCvFlann, KnnCt)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 1})
    ->Args({6, 1})
    ->Args({8, 1})
    ->Args({10, 1})
    ->Args({12, 1})
    ->Args({14, 1})
    ->Args({1, 4})
    ->Args({6, 4})
    ->Args({8, 4})
    ->Args({10, 4})
    ->Args({12, 4})
    ->Args({14, 4})
    ->Args({1, 8})
    ->Args({6, 8})
    ->Args({8, 8})
    ->Args({10, 8})
    ->Args({12, 8})
    ->Args({14, 8})
    ->Args({1, 12})
    ->Args({6, 12})
    ->Args({8, 12})
    ->Args({10, 12})
    ->Args({12, 12})
    ->Args({14, 12});

BENCHMARK_MAIN();
