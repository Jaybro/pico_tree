#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/kd_tree_builder.hpp>
#include <pico_tree/internal/space_wrapper.hpp>
#include <pico_tree/vector_traits.hpp>

namespace {

template <typename Point_>
using space = std::reference_wrapper<std::vector<Point_>>;

}  // namespace

TEST(KdTreeTest, SplitterMedian) {
  using point_type = pico_tree::point_2f;
  using index_type = std::size_t;
  using scalar_type = typename point_type::scalar_type;
  using space_type = space<point_type>;
  using splitter_type = pico_tree::internal::splitter_median_max_side<
      pico_tree::internal::space_wrapper<space_type>>;

  // Check split of a list with an even number of elements.
  std::vector<point_type> ptsx4{
      {0.0f, 4.0f}, {0.0f, 2.0f}, {0.0f, 3.0f}, {0.0f, 1.0f}};
  space_type spcx4(ptsx4);
  pico_tree::internal::space_wrapper<space_type> spcx4_wrapper(spcx4);
  std::vector<index_type> idx4{0, 1, 2, 3};

  pico_tree::internal::box<scalar_type, 2> box(2);
  box.min(0) = 0.0f;
  box.min(1) = 0.0f;
  box.max(0) = 1.0f;
  box.max(1) = 0.0f;
  std::vector<index_type>::iterator split;
  pico_tree::size_t split_dim;
  scalar_type split_val;

  splitter_type splitter4(spcx4_wrapper);
  splitter4(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[2][0]);

  // Check split of a list with an odd number of elements.
  std::vector<point_type> ptsx7{
      {3.0f, 6.0f},
      {0.0f, 4.0f},
      {0.0f, 2.0f},
      {0.0f, 5.0f},
      {0.0f, 3.0f},
      {0.0f, 1.0f},
      {1.0f, 7.0f}};
  space_type spcx7(ptsx7);
  pico_tree::internal::space_wrapper<space_type> spcx7_wrapper(spcx7);
  std::vector<index_type> idx7{0, 1, 2, 3, 4, 5, 6};

  splitter_type splitter7(spcx7_wrapper);
  splitter7(0, idx7.begin(), idx7.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx7.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx7[idx7[3]][0]);

  box.max(1) = 10.0f;
  splitter7(1, idx7.begin() + 3, idx7.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx7.begin(), 5);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx7[idx7[5]][1]);
}

TEST(KdTreeTest, SplitterMidpoint) {
  using point_type = pico_tree::point_2f;
  using index_type = int;
  using scalar_type = typename point_type::scalar_type;
  using space_type = space<point_type>;
  using splitter_type = pico_tree::internal::splitter_midpoint_max_side<
      pico_tree::internal::space_wrapper<space_type>>;

  std::vector<point_type> ptsx4{{0.0, 2.0}, {0.0, 1.0}, {0.0, 4.0}, {0.0, 3.0}};
  space_type spcx4(ptsx4);
  pico_tree::internal::space_wrapper<space_type> spcx4_wrapper(spcx4);
  std::vector<index_type> idx4{0, 1, 2, 3};

  splitter_type splitter(spcx4_wrapper);

  pico_tree::internal::box<scalar_type, 2> box(2);
  std::vector<index_type>::iterator split;
  pico_tree::size_t split_dim;
  scalar_type split_val;

  // Everything is forced to the right leaf. This means that the split must be
  // on index 0.
  box.min(0) = scalar_type{0.0};
  box.min(1) = scalar_type{0.0};
  box.max(0) = scalar_type{0.0};
  box.max(1) = scalar_type{1.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 0);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / scalar_type{2.0});

  // Everything is forced to the left leaf. This means that the split must be on
  // index 4.
  box.max(1) = scalar_type{9.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 4);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / scalar_type{2.0});

  // Clean middle split. A general case where the split value falls somewhere
  // in between the range of numbers.
  box.max(1) = scalar_type{5.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / scalar_type{2.0});

  // On dimension 0 we test what happens when all values are equal. Again
  // everything moves to the left. So we want to split on index 4.
  box.max(0) = scalar_type{15.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 4);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, (box.max(0) + box.min(0)) / scalar_type{2.0});
}

TEST(KdTreeTest, SplitterSlidingMidpoint) {
  using point_type = pico_tree::point_2f;
  using index_type = int;
  using scalar_type = typename point_type::scalar_type;
  using space_type = space<point_type>;
  using splitter_type = pico_tree::internal::splitter_sliding_midpoint_max_side<
      pico_tree::internal::space_wrapper<space_type>>;

  std::vector<point_type> ptsx4{{0.0, 2.0}, {0.0, 1.0}, {0.0, 4.0}, {0.0, 3.0}};
  space_type spcx4(ptsx4);
  pico_tree::internal::space_wrapper<space_type> spcx4_wrapper(spcx4);
  std::vector<index_type> idx4{0, 1, 2, 3};

  splitter_type splitter(spcx4_wrapper);

  pico_tree::internal::box<scalar_type, 2> box(2);
  std::vector<index_type>::iterator split;
  pico_tree::size_t split_dim;
  scalar_type split_val;

  // Everything is forced to the right leaf. This means we want a single point
  // to the left (the lowest value) and internally the splitter needs to reorder
  // the indices such that on index 1 we get value 2.
  box.min(0) = scalar_type{0.0};
  box.min(1) = scalar_type{0.0};
  box.max(0) = scalar_type{0.0};
  box.max(1) = scalar_type{1.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 1);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[0][1]);
  EXPECT_EQ(idx4[0], 1);
  EXPECT_EQ(idx4[1], 0);

  // Everything is forced to the left leaf. This means we want a single point
  // to the right (the highest value) and internally the splitter needs to
  // reorder the indices such that on index 3 we get value 4.
  box.max(1) = scalar_type{9.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, ptsx4[2][1]);
  EXPECT_EQ(idx4[3], 2);

  // Clean middle split. A general case where the split value falls somewhere
  // in between the range of numbers.
  box.max(1) = scalar_type{5.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 2);
  EXPECT_EQ(split_dim, 1);
  EXPECT_EQ(split_val, (box.max(1) + box.min(1)) / scalar_type{2.0});

  // On dimension 0 we test what happens when all values are equal. Again
  // everything moves to the left. So we want to split on index 3.
  box.max(0) = scalar_type{15.0};
  splitter(0, idx4.begin(), idx4.end(), box, split, split_dim, split_val);

  EXPECT_EQ(split - idx4.begin(), 3);
  EXPECT_EQ(split_dim, 0);
  EXPECT_EQ(split_val, ptsx4[3][0]);
}
