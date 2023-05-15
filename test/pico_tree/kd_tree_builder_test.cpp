#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/kd_tree_builder.hpp>
#include <pico_tree/internal/space_wrapper.hpp>

namespace {

template <typename PointX>
using Space = std::reference_wrapper<std::vector<PointX>>;

template <typename SpaceX>
using Traits = pico_tree::StdTraits<SpaceX>;

}  // namespace

TEST(KdTreeTest, SplitterMedian) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using SpaceX = Space<PointX>;
  using SplitterX = pico_tree::internal::SplitterLongestMedian<
      pico_tree::internal::SpaceWrapper<Traits<SpaceX>>>;

  // Check split of a list with an even amount of elements.
  std::vector<PointX> ptsx4{
      {0.0f, 4.0f}, {0.0f, 2.0f}, {0.0f, 3.0f}, {0.0f, 1.0f}};
  SpaceX spcx4(ptsx4);
  pico_tree::internal::SpaceWrapper<Traits<SpaceX>> spcx4_wrapper(spcx4);
  std::vector<Index> idx4{0, 1, 2, 3};

  pico_tree::internal::Box<Scalar, 2> box(2);
  box.min(0) = 0.0f;
  box.min(1) = 0.0f;
  box.max(0) = 1.0f;
  box.max(1) = 0.0f;
  std::vector<Index>::iterator split;
  pico_tree::Size split_dim;
  Scalar split_val;

  SplitterX splitter4(spcx4_wrapper);
  splitter4(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[2](0));

  // Check split of a list with an odd amount of elements.
  std::vector<PointX> ptsx7{
      {3.0f, 6.0f},
      {0.0f, 4.0f},
      {0.0f, 2.0f},
      {0.0f, 5.0f},
      {0.0f, 3.0f},
      {0.0f, 1.0f},
      {1.0f, 7.0f}};
  SpaceX spcx7(ptsx7);
  pico_tree::internal::SpaceWrapper<Traits<SpaceX>> spcx7_wrapper(spcx7);
  std::vector<Index> idx7{0, 1, 2, 3, 4, 5, 6};

  SplitterX splitter7(spcx7_wrapper);
  splitter7(0, idx7.begin(), idx7.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx7.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx7[idx7[3]](0));

  box.max(1) = 10.0f;
  splitter7(1, idx7.begin() + 3, idx7.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx7.begin(), 5);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx7[idx7[5]](1));
}

TEST(KdTreeTest, SplitterSlidingMidpoint) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using SpaceX = Space<PointX>;
  using SplitterX = pico_tree::internal::SplitterSlidingMidpoint<
      pico_tree::internal::SpaceWrapper<Traits<SpaceX>>>;

  std::vector<PointX> ptsx4{{0.0, 2.0}, {0.0, 1.0}, {0.0, 4.0}, {0.0, 3.0}};
  SpaceX spcx4(ptsx4);
  pico_tree::internal::SpaceWrapper<Traits<SpaceX>> spcx4_wrapper(spcx4);
  std::vector<Index> idx4{0, 1, 2, 3};

  SplitterX splitter(spcx4_wrapper);

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
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 1);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[0](1));
  EXPECT_EQ(idx4[0], 1);
  EXPECT_EQ(idx4[1], 0);

  // Everything is forced to the left leaf. This means we want a single point
  // to the right (the highest value) and internally the splitter needs to
  // reorder the indices such that on index 3 we get value 4.
  box.max(1) = Scalar{9.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[2](1));
  EXPECT_EQ(idx4[3], 2);

  // Clean middle split. A general case where the split value falls somewhere
  // inbetween the range of numbers.
  box.max(1) = Scalar{5.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / Scalar{2.0});

  // On dimension 0 we test what happens when all values are equal. Again
  // everything moves to the left. So we want to split on index 3.
  box.max(0) = Scalar{15.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[3](0));
}
