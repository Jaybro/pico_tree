#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <pico_tree/map.hpp>

using namespace pico_tree;

namespace {

Size constexpr kDynamicMapDim = 4;

constexpr Size Dim(Size dim) {
  return dim != kDynamicSize ? dim : kDynamicMapDim;
}

}  // namespace

template <typename T>
class PointMapTest : public testing::Test {};

template <typename Scalar_, Size Dim_>
class PointMapTest<PointMap<Scalar_, Dim_>> : public testing::Test {
 public:
  static constexpr Size Sdim = Dim(Dim_);

  PointMapTest() : map_(scalars_.data(), Sdim) {
    std::iota(scalars_.begin(), scalars_.end(), Scalar_(0.0));
  }

 protected:
  std::array<Scalar_, Sdim> scalars_;
  PointMap<Scalar_, Dim_> map_;
};

using PointMapTypes = testing::Types<
    PointMap<float, 2>,
    PointMap<double, 3>,
    PointMap<float, kDynamicSize>>;

TYPED_TEST_SUITE(PointMapTest, PointMapTypes);

TYPED_TEST(PointMapTest, Accessors) {
  using Scalar = typename TypeParam::ScalarType;

  for (Size i = 0; i < this->map_.size(); ++i) {
    EXPECT_EQ(this->map_[i], Scalar(i));
  }
}

TYPED_TEST(PointMapTest, size) {
  EXPECT_EQ(this->map_.size(), this->scalars_.size());
}
