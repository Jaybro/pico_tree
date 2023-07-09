#include <gtest/gtest.h>

#include <numeric>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/map.hpp>

using namespace pico_tree;

namespace {

Size constexpr kDynamicMapDim = 4;

constexpr Size Dim(Size dim) {
  return dim != kDynamicSize ? dim : kDynamicMapDim;
}

}  // namespace

template <typename T>
class SpaceMapTest : public testing::Test {};

template <typename Point_>
class SpaceMapTest<SpaceMap<Point_>> : public testing::Test {
 public:
  SpaceMapTest() : map_(points_.data(), points_.size()) {
    std::size_t count = 0;
    for (auto& point : points_) {
      for (auto& coord : point) {
        coord = typename SpaceMap<Point_>::ScalarType(count);
        count++;
      }
    }
  }

  Point_ const& PointAt(std::size_t index) const { return points_[index]; }

  std::size_t size() const { return points_.size(); }

  constexpr std::size_t sdim() const { return PointTraits<Point_>::Dim; }

 protected:
  std::array<Point_, 2> points_;
  SpaceMap<Point_> map_;
};

template <typename Scalar_, Size Dim_>
class SpaceMapTest<SpaceMap<PointMap<Scalar_, Dim_>>> : public testing::Test {
 public:
  static constexpr Size Sdim = Dim(Dim_);

  SpaceMapTest() : map_(coords_.data(), coords_.size() / Sdim, Sdim) {
    std::size_t count = 0;
    for (auto& coord : coords_) {
      coord = Scalar_(count);
      count++;
    }
  }

  PointMap<Scalar_ const, Dim_> PointAt(std::size_t index) const {
    return {coords_.data() + index * Sdim, Sdim};
  }

  std::size_t size() const { return coords_.size() / Sdim; }

  constexpr std::size_t sdim() const { return Sdim; }

 protected:
  std::array<Scalar_, 2 * Sdim> coords_;
  SpaceMap<PointMap<Scalar_, Dim_>> map_;
};

using SpaceMapTypes = testing::Types<
    SpaceMap<std::array<float, 2>>,
    SpaceMap<std::array<double, 3>>,
    SpaceMap<PointMap<float, 2>>,
    SpaceMap<PointMap<double, kDynamicSize>>>;

TYPED_TEST_SUITE(SpaceMapTest, SpaceMapTypes);

TYPED_TEST(SpaceMapTest, Accessors) {
  for (Size i = 0; i < this->map_.size(); ++i) {
    EXPECT_EQ(this->map_[i].data(), this->PointAt(i).data());
  }
}

TYPED_TEST(SpaceMapTest, size) { EXPECT_EQ(this->map_.size(), this->size()); }

TYPED_TEST(SpaceMapTest, sdim) { EXPECT_EQ(this->map_.sdim(), this->sdim()); }
