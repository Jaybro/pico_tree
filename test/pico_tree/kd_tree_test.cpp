#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/std_traits.hpp>

#include "common.hpp"

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename PointX>
using Space = std::reference_wrapper<std::vector<PointX>>;

template <typename SpaceX>
using Traits = pico_tree::StdTraits<SpaceX>;

template <typename PointX>
using KdTree = pico_tree::KdTree<Traits<Space<PointX>>>;

// Wraps a vector and provides a dynamic spatial dimension.
template <typename Point>
class DynamicSpace {
 public:
  using SizeType = pico_tree::Size;

  DynamicSpace(std::vector<Point> const& space, SizeType sdim)
      : space_(space), sdim_(sdim) {}

  inline operator std::vector<Point> const&() const { return space_; }

  SizeType sdim() const { return sdim_; }

 private:
  std::vector<Point> const& space_;
  SizeType sdim_;
};

// Supports a dynamic spatial dimension for vectors.
template <typename Point>
struct DynamicSpaceTraits : public pico_tree::StdTraits<std::vector<Point>> {
  using SpaceType = DynamicSpace<Point>;
  using SizeType = pico_tree::Size;
  static SizeType constexpr Dim = pico_tree::kDynamicSize;

  inline static SizeType SpaceSdim(DynamicSpace<Point> const& space) {
    return space.sdim();
  }
};

template <typename PointX>
void QueryRange(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const min_v,
    typename PointX::ScalarType const max_v) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree(random, 8);

  TestBox(tree, min_v, max_v);
}

template <typename PointX>
void QueryRadius(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const radius) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree(random, 8);

  TestRadius(tree, radius);
}

template <typename PointX>
void QueryKnn(
    int const point_count,
    typename PointX::ScalarType const area_size,
    int const k) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree1(random, 8);

  // "Test" move constructor.
  auto tree2 = std::move(tree1);
  // "Test" move assignment.
  tree1 = std::move(tree2);

  TestKnn(tree1, static_cast<typename KdTree<PointX>::IndexType>(k));
}

}  // namespace

TEST(KdTreeTest, QueryRangeSubset2d) {
  QueryRange<Point2f>(1024 * 1024, 100.0f, 15.1f, 34.9f);
}

TEST(KdTreeTest, QueryRangeAll2d) {
  QueryRange<Point2f>(1024, 10.0f, 0.0f, 10.0f);
}

TEST(KdTreeTest, QueryRadiusSubset2d) {
  QueryRadius<Point2f>(1024 * 1024, 100.0f, 2.5f);
}

TEST(KdTreeTest, QueryKnn1) { QueryKnn<Point2f>(1024 * 1024, 100.0f, 1); }

TEST(KdTreeTest, QueryKnn10) { QueryKnn<Point2f>(1024 * 1024, 100.0f, 10); }

TEST(KdTreeTest, QuerySo2Knn4) {
  using PointX = Point1f;
  using TraitsX = Traits<Space<PointX>>;

  const auto pi = pico_tree::internal::kPi<typename KdTree<PointX>::ScalarType>;
  std::vector<PointX> random = GenerateRandomN<PointX>(256 * 256, -pi, pi);
  pico_tree::KdTree<TraitsX, pico_tree::SO2> tree(random, 10);
  TestKnn(tree, static_cast<typename KdTree<PointX>::IndexType>(8), PointX{pi});
}

TEST(KdTreeTest, WriteRead) {
  using Index = int;
  using Scalar = typename Point2f::ScalarType;
  Index point_count = 100;
  Scalar area_size = 2;
  std::vector<Point2f> random =
      GenerateRandomN<Point2f>(point_count, area_size);

  std::string filename = "tree.bin";

  // Compile time known dimensions.
  {
    // The points are not stored.
    KdTree<Point2f> tree(random, 1);
    KdTree<Point2f>::Save(tree, filename);
  }
  {
    // Points are required to load the tree.
    KdTree<Point2f> tree = KdTree<Point2f>::Load(random, filename);
    TestKnn(tree, Index(20));
  }

  EXPECT_EQ(std::remove(filename.c_str()), 0);

  // Run time known dimensions.
  using DSpace = DynamicSpace<Point2f>;
  using DTraits = DynamicSpaceTraits<Point2f>;

  DSpace drandom(random, 2);

  {
    // The points are not stored.
    pico_tree::KdTree<DTraits> tree(drandom, 1);
    pico_tree::KdTree<DTraits>::Save(tree, filename);
  }
  {
    // Points are required to load the tree.
    pico_tree::KdTree<DTraits> tree =
        pico_tree::KdTree<DTraits>::Load(drandom, filename);
    TestKnn(tree, Index(20));
  }

  EXPECT_EQ(std::remove(filename.c_str()), 0);
}
