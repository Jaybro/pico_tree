#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>

template <int Dim, typename Space>
void StdCheckTypes(Space const& space, std::size_t npts) {
  static_assert(
      std::is_same<typename pico_tree::StdTraits<Space>::SpaceType, Space>::
          value,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(
      pico_tree::StdTraits<Space>::Dim == Dim, "TRAITS_DIM_INCORRECT");

  static_assert(
      std::is_same<typename pico_tree::StdTraits<Space>::IndexType, int>::value,
      "TRAITS_INDEX_TYPE_NOT_INT");

  static_assert(
      std::is_same<
          typename pico_tree::StdTraits<Space, std::size_t>::IndexType,
          std::size_t>::value,
      "TRAITS_INDEX_TYPE_NOT_SIZE_T");

  EXPECT_EQ(Dim, pico_tree::StdTraits<Space>::SpaceSdim(space));
  EXPECT_EQ(
      static_cast<int>(npts), pico_tree::StdTraits<Space>::SpaceNpts(space));

  Point2f const& p = pico_tree::StdTraits<Space>::PointAt(space, 0);
  EXPECT_FLOAT_EQ(p(0), 1.0f);
  EXPECT_FLOAT_EQ(p(1), 2.0f);
}

std::vector<Point2f> GetStdVector() { return {{1.0f, 2.0f}}; }

TEST(StdTraitsTest, StdVector) {
  using Space = std::vector<Point2f>;

  std::vector<Point2f> points = GetStdVector();
  StdCheckTypes<Point2f::Dim, Space>(points, points.size());
}

TEST(StdTraitsTest, StdRefVector) {
  using Space = std::reference_wrapper<std::vector<Point2f>>;

  std::vector<Point2f> points = GetStdVector();
  StdCheckTypes<Point2f::Dim, Space>(std::ref(points), points.size());
}
