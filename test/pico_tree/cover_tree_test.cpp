#include <gtest/gtest.h>

#include <pico_toolshed/pico_adaptor.hpp>
#include <pico_understory/cover_tree.hpp>

#include "common.hpp"

template <typename PicoAdaptor>
using CoverTree = pico_tree::CoverTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    PicoAdaptor::Dim,
    PicoAdaptor>;

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename PointX>
void QueryRadius(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const radius) {
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using AdaptorX = PicoAdaptor<Index, PointX>;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  AdaptorX adaptor(random);
  CoverTree<AdaptorX> tree(adaptor, Scalar(2.0));

  TestRadius(tree, radius);
}

template <typename PointX>
void QueryKnn(
    int const point_count,
    typename PointX::ScalarType const area_size,
    int const k) {
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using AdaptorX = PicoAdaptor<Index, PointX>;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  AdaptorX adaptor(random);
  CoverTree<AdaptorX> tree(adaptor, Scalar(2.0));

  // This line compile time "tests" the move capability of the tree.
  auto tree2 = std::move(tree);

  TestKnn(tree2, static_cast<Index>(k));
}

}  // namespace

TEST(CoverTreeTest, QueryRadiusSubset2d) {
  QueryRadius<Point2f>(1024 * 128, 100.0f, 2.5f);
}

TEST(CoverTreeTest, QueryKnn1) { QueryKnn<Point2f>(1024 * 128, 100.0f, 1); }

TEST(CoverTreeTest, QueryKnn10) { QueryKnn<Point2f>(1024 * 128, 100.0f, 10); }
