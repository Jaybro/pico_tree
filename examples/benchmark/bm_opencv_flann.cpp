#include <opencv2/flann.hpp>

#include "benchmark.hpp"

// BmOpenCvFlann benchmarks the OpenCV version of FLANN. It is possible to
// simply include the original FLANN <flann/flann.hpp> and replace the cvflann
// namespace by the flann namespace and the benchmark will still work.
// NOTE: The OpenCV version of FLANN performs quite a bit faster for queries
// than https://github.com/mariusmuja/flann. About 30% for a single NN. Tree
// build times are the same. Simply used Release for building like with all
// other libs.
class BmOpenCvFlann : public pico_tree::Benchmark {
 public:
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmOpenCvFlann, BuildRt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  cvflann::Matrix<Scalar> matrix(points_.data()->data, points_.size(), 3);
  // Reorder will change the order of the input to fit the generated indices,
  // but it will replace (delete) the original input.
  cvflann::KDTreeSingleIndexParams pindex(max_leaf_size, false);

  for (auto _ : state) {
    cvflann::KDTreeSingleIndex<cvflann::L2_Simple<Scalar>> tree(matrix, pindex);
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

BENCHMARK_DEFINE_F(BmOpenCvFlann, KnnRt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);

  cvflann::Matrix<Scalar> matrix(points_.data()->data, points_.size(), 3);
  // Reorder will change the order of the input to fit the generated indices,
  // but it will replace (and delete) the original input. Note that the reorder
  // option does not appear to have effect on the performance of the queries.
  cvflann::KDTreeSingleIndexParams pindex(max_leaf_size, false);
  // It seems that there are different versions of FLANN. For example, the
  // OpenCV version does not have the flann::L2_3D<Scalar> distance class, like
  // found on the following GitHub respository:
  // https://github.com/mariusmuja/flann/blob/master/src/cpp/flann/algorithms/dist.h
  // However, using the exact same one or a custom metric where the dimensions
  // are know at compile time does not appear to really impact the performance.
  cvflann::KDTreeSingleIndex<cvflann::L2_Simple<Scalar>> tree(matrix, pindex);
  tree.buildIndex();

  // Search all nodes, no approximate search and no sorting.
  cvflann::SearchParams psearch(-1, 0.0f, false);

  // There is also the option to query them all at once, but this doesn't really
  // change performance and this version looks more like the other benchmarks.
  for (auto _ : state) {
    std::vector<Index> indices(knn_count);
    std::vector<Scalar> distances(knn_count);
    cvflann::Matrix<Index> mat_indices(indices.data(), 1, knn_count);
    cvflann::Matrix<Scalar> mat_distances(distances.data(), 1, knn_count);

    for (auto& p : points_) {
      cvflann::Matrix<Scalar> query(p.data, 1, 3);
      tree.knnSearch(query, mat_indices, mat_distances, knn_count, psearch);
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: K nearest neighbors.
BENCHMARK_REGISTER_F(BmOpenCvFlann, KnnRt)
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
