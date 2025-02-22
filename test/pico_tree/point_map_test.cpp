#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <pico_tree/map.hpp>

using namespace pico_tree;

namespace {

size_t constexpr dynamic_map_dim = 4;

constexpr size_t dimension(size_t d) {
  return d != dynamic_size ? d : dynamic_map_dim;
}

}  // namespace

template <typename T>
class PointMapTest : public testing::Test {};

template <typename Scalar_, size_t Dim_>
class PointMapTest<point_map<Scalar_, Dim_>> : public testing::Test {
 public:
  static constexpr size_t dim = dimension(Dim_);

  PointMapTest() : map_(scalars_.data(), dim) {
    std::iota(scalars_.begin(), scalars_.end(), Scalar_(0.0));
  }

 protected:
  std::array<Scalar_, dim> scalars_;
  point_map<Scalar_, Dim_> map_;
};

using PointMapTypes = testing::Types<
    point_map<float, 2>,
    point_map<double, 3>,
    point_map<float, dynamic_size>>;

TYPED_TEST_SUITE(PointMapTest, PointMapTypes);

TYPED_TEST(PointMapTest, Accessors) {
  using scalar_type = typename TypeParam::scalar_type;

  for (size_t i = 0; i < this->map_.size(); ++i) {
    EXPECT_EQ(this->map_[i], scalar_type(i));
  }
}

TYPED_TEST(PointMapTest, size) {
  EXPECT_EQ(this->map_.size(), this->scalars_.size());
}
