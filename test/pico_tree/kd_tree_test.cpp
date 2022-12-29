#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>

#include "common.hpp"

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
  static SizeType constexpr Dim = pico_tree::kDynamicDim;

  inline static SizeType SpaceSdim(DynamicSpace<Point> const& space) {
    return space.sdim();
  }
};

}  // namespace

TEST(KdTreeTest, SplitterMedian) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using SpaceX = Space<PointX>;
  using SplitterX = pico_tree::SplitterLongestMedian<Traits<SpaceX>>;

  std::vector<PointX> ptsx4{
      {0.0f, 4.0f}, {0.0f, 2.0f}, {0.0f, 3.0f}, {0.0f, 1.0f}};
  SpaceX spcx4(ptsx4);
  std::vector<Index> idx4{0, 1, 2, 3};

  pico_tree::internal::Box<Scalar, 2> box(2);
  box.min(0) = 0.0f;
  box.min(1) = 0.0f;
  box.max(0) = 1.0f;
  box.max(1) = 0.0f;
  std::vector<Index>::iterator split;
  pico_tree::Size split_dim;
  Scalar split_val;

  SplitterX splitter4(spcx4);
  splitter4(0, idx4.begin(), idx4.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[2](0));

  std::vector<PointX> ptsx7{
      {3.0f, 6.0f},
      {0.0f, 4.0f},
      {0.0f, 2.0f},
      {0.0f, 5.0f},
      {0.0f, 3.0f},
      {0.0f, 1.0f},
      {1.0f, 7.0f}};
  SpaceX spcx7(ptsx7);
  std::vector<Index> idx7{0, 1, 2, 3, 4, 5, 6};

  SplitterX splitter7(spcx7);
  splitter7(0, idx7.begin(), idx7.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx7.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx7[idx7[3]](0));

  box.max(1) = 10.0f;
  splitter7(
      1, idx7.begin() + 3, idx7.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx7.begin(), 5);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx7[idx7[5]](1));
}

TEST(KdTreeTest, SplitterSlidingMidpoint) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using SpaceX = Space<PointX>;
  using SplitterX = pico_tree::SplitterSlidingMidpoint<Traits<SpaceX>>;

  std::vector<PointX> ptsx4{{0.0, 2.0}, {0.0, 1.0}, {0.0, 4.0}, {0.0, 3.0}};
  SpaceX spcx4(ptsx4);
  std::vector<Index> idx4{0, 1, 2, 3};

  SplitterX splitter(spcx4);

  pico_tree::internal::Box<Scalar, 2> box(2);
  std::vector<Index>::iterator split;
  pico_tree::Size split_dim;
  Scalar split_val;

  // Everything is forced to the right leaf. This means we want a single point
  // to the left (the lowest value) and internally the splitter needs to reorder
  // the indices such that on index 1 we get value 2.
  box.min(0) = Scalar{0.0};
  box.min(1) = Scalar{0.0};
  box.max(0) = Scalar{0.0};
  box.max(1) = Scalar{1.0};
  splitter(0, idx4.begin(), idx4.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx4.begin(), 1);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[0](1));
  EXPECT_EQ(idx4[0], 1);
  EXPECT_EQ(idx4[1], 0);

  // Everything is forced to the left leaf. This means we want a single point
  // to the right (the highest value) and internally the splitter needs to
  // reorder the indices such that on index 3 we get value 4.
  box.max(1) = Scalar{9.0};
  splitter(0, idx4.begin(), idx4.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[2](1));
  EXPECT_EQ(idx4[3], 2);

  // Clean middle split. A general case where the split value falls somewhere
  // inbetween the range of numbers.
  box.max(1) = Scalar{5.0};
  splitter(0, idx4.begin(), idx4.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / Scalar{2.0});

  // On dimension 0 we test what happens when all values are equal. Again
  // everything moves to the left. So we want to split on index 3.
  box.max(0) = Scalar{15.0};
  splitter(0, idx4.begin(), idx4.end(), box, &split, &split_dim, &split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[3](0));
}

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

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

  const auto pi = typename KdTree<PointX>::ScalarType(pico_tree::internal::kPi);
  std::vector<PointX> random = GenerateRandomN<PointX>(256 * 256, -pi, pi);
  pico_tree::KdTree<TraitsX, pico_tree::SO2<TraitsX>> tree(random, 10);
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
