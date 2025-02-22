#include <gtest/gtest.h>

#include <pico_tree/internal/box.hpp>

using namespace pico_tree;
using namespace pico_tree::internal;

namespace {

constexpr size_t dynamic_box_dim = 4;

inline size_t constexpr dimension(size_t d) {
  return d != dynamic_size ? d : dynamic_box_dim;
}

}  // namespace

template <typename T>
class BoxTest : public testing::Test {};

template <typename Scalar_, size_t Dim_>
class BoxTest<box<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest() : box_(dimension(Dim_)) {}

 protected:
  box<Scalar_, Dim_> box_;
};

template <typename Scalar_, size_t Dim_>
class BoxTest<box_map<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest()
      : min_(dimension(Dim_)),
        max_(dimension(Dim_)),
        box_(min_.data(), max_.data(), dimension(Dim_)) {}

 protected:
  std::vector<Scalar_> min_;
  std::vector<Scalar_> max_;
  box_map<Scalar_, Dim_> box_;
};

using BoxTestTypes = testing::Types<
    box<float, 2>,
    box<double, 3>,
    box<float, dynamic_size>,
    box_map<float, 2>,
    box_map<double, 3>,
    box_map<float, dynamic_size>>;

TYPED_TEST_SUITE(BoxTest, BoxTestTypes);

TYPED_TEST(BoxTest, size) {
  size_t dim = TypeParam::dim;
  if (dim != dynamic_size) {
    EXPECT_EQ(this->box_.size(), dim);
  } else {
    EXPECT_EQ(this->box_.size(), dynamic_box_dim);
  }
}

TYPED_TEST(BoxTest, Accessors) {
  this->box_.fill_inverse_max();
  for (size_t i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), this->box_.min()[i]);
    EXPECT_EQ(this->box_.max(i), this->box_.max()[i]);
  }
}

TYPED_TEST(BoxTest, FillInverseMax) {
  using scalar_type = typename TypeParam::scalar_type;

  this->box_.fill_inverse_max();
  for (size_t i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), std::numeric_limits<scalar_type>::max());
    EXPECT_EQ(this->box_.max(i), std::numeric_limits<scalar_type>::lowest());
  }
}

TYPED_TEST(BoxTest, Fit) {
  using scalar_type = typename TypeParam::scalar_type;

  this->box_.fill_inverse_max();
  size_t dim = dimension(TypeParam::dim);

  // Fit a point.
  std::vector<scalar_type> p(dim, scalar_type(0));
  this->box_.fit(p.data());
  for (size_t i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), scalar_type(0));
    EXPECT_EQ(this->box_.max(i), scalar_type(0));
  }

  // Fit a box.
  std::vector<scalar_type> min(dim, scalar_type(-1));
  std::vector<scalar_type> max(dim, scalar_type(+1));
  this->box_.fit(
      box_map<scalar_type, dynamic_size>(min.data(), max.data(), dim));
  for (size_t i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), min[i]);
    EXPECT_EQ(this->box_.max(i), max[i]);
  }
}
