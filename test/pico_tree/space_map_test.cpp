#include <gtest/gtest.h>

#include <numeric>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/map.hpp>

using namespace pico_tree;

namespace {

size_t constexpr dynamic_map_dim = 4;

constexpr size_t dimension(size_t d) {
  return d != dynamic_extent ? d : dynamic_map_dim;
}

}  // namespace

template <typename T_>
class SpaceMapTest : public testing::Test {};

template <typename Point_>
class SpaceMapTest<space_map<Point_>> : public testing::Test {
 public:
  SpaceMapTest() : map_(points_.data(), points_.size()) {
    std::size_t count = 0;
    for (auto& point : points_) {
      for (auto& coord : point) {
        coord = typename space_map<Point_>::scalar_type(count);
        count++;
      }
    }
  }

  Point_ const& point_at(std::size_t index) const { return points_[index]; }

  std::size_t size() const { return points_.size(); }

  constexpr std::size_t sdim() const { return point_traits<Point_>::dim; }

 protected:
  std::array<Point_, 2> points_;
  space_map<Point_> map_;
};

template <typename Scalar_, size_t Dim_>
class SpaceMapTest<space_map<point_map<Scalar_, Dim_>>> : public testing::Test {
 public:
  static constexpr size_t dim = dimension(Dim_);

  SpaceMapTest() : map_(coords_.data(), coords_.size() / dim, dim) {
    std::size_t count = 0;
    for (auto& coord : coords_) {
      coord = Scalar_(count);
      count++;
    }
  }

  point_map<Scalar_ const, Dim_> point_at(std::size_t index) const {
    return {coords_.data() + index * dim, dim};
  }

  std::size_t size() const { return coords_.size() / dim; }

  constexpr std::size_t sdim() const { return dim; }

 protected:
  std::array<Scalar_, 2 * dim> coords_;
  space_map<point_map<Scalar_, Dim_>> map_;
};

using SpaceMapTypes = testing::Types<
    space_map<std::array<float, 2>>,
    space_map<std::array<double, 3>>,
    space_map<point_map<float, 2>>,
    space_map<point_map<double, dynamic_extent>>>;

TYPED_TEST_SUITE(SpaceMapTest, SpaceMapTypes);

TYPED_TEST(SpaceMapTest, Accessors) {
  for (size_t i = 0; i < this->map_.size(); ++i) {
    EXPECT_EQ(this->map_[i].data(), this->point_at(i).data());
  }
}

TYPED_TEST(SpaceMapTest, size) { EXPECT_EQ(this->map_.size(), this->size()); }

TYPED_TEST(SpaceMapTest, sdim) { EXPECT_EQ(this->map_.sdim(), this->sdim()); }
