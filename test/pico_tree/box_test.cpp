#include <gtest/gtest.h>

#include <pico_tree/internal/box.hpp>

using namespace pico_tree;
using namespace pico_tree::internal;

Size constexpr kDynamicBoxDim = 4;

inline Size constexpr Dim(Size dim) {
  return dim != kDynamicDim ? dim : kDynamicBoxDim;
}

template <typename T>
class BoxTest : public testing::Test {};

template <typename Scalar_, Size Dim_>
class BoxTest<Box<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest() : box_(Dim(Dim_)) {}

 protected:
  Box<Scalar_, Dim_> box_;
};

template <typename Scalar_, Size Dim_>
class BoxTest<BoxMap<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest()
      : min_(Dim(Dim_)),
        max_(Dim(Dim_)),
        box_(min_.data(), max_.data(), Dim(Dim_)) {}

 protected:
  std::vector<Scalar_> min_;
  std::vector<Scalar_> max_;
  BoxMap<Scalar_, Dim_> box_;
};

using BoxTestTypes = testing::Types<
    Box<float, 2>,
    Box<double, 3>,
    Box<float, kDynamicDim>,
    BoxMap<float, 2>,
    BoxMap<double, 3>,
    BoxMap<float, kDynamicDim>>;

TYPED_TEST_SUITE(BoxTest, BoxTestTypes);

TYPED_TEST(BoxTest, size) {
  Size dim = TypeParam::Dim;
  if (dim != kDynamicDim) {
    EXPECT_EQ(this->box_.size(), dim);
  } else {
    EXPECT_EQ(this->box_.size(), kDynamicBoxDim);
  }
}

TYPED_TEST(BoxTest, Accessors) {
  this->box_.FillInverseMax();
  for (Size i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), this->box_.min()[i]);
    EXPECT_EQ(this->box_.max(i), this->box_.max()[i]);
  }
}

TYPED_TEST(BoxTest, FillInverseMax) {
  using Scalar = typename TypeParam::ScalarType;

  this->box_.FillInverseMax();
  for (Size i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), std::numeric_limits<Scalar>::max());
    EXPECT_EQ(this->box_.max(i), std::numeric_limits<Scalar>::lowest());
  }
}

TYPED_TEST(BoxTest, Fit) {
  using Scalar = typename TypeParam::ScalarType;

  this->box_.FillInverseMax();
  Size dim = Dim(TypeParam::Dim);

  // Fit a point.
  std::vector<Scalar> p(dim, Scalar(0));
  this->box_.Fit(p.data());
  for (Size i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), Scalar(0));
    EXPECT_EQ(this->box_.max(i), Scalar(0));
  }

  // Fit a box.
  std::vector<Scalar> min(dim, Scalar(-1));
  std::vector<Scalar> max(dim, Scalar(+1));
  this->box_.Fit(BoxMap<Scalar, kDynamicDim>(min.data(), max.data(), dim));
  for (Size i = 0; i < this->box_.size(); ++i) {
    EXPECT_EQ(this->box_.min(i), min[i]);
    EXPECT_EQ(this->box_.max(i), max[i]);
  }
}
