#include <gtest/gtest.h>

#include <pico_tree/internal/box.hpp>

using namespace pico_tree;
using namespace pico_tree::internal;

int constexpr kDynamicBoxDim = 4;

inline constexpr int Dim(int dim) {
  return dim != kDynamicDim ? dim : kDynamicBoxDim;
}

template <typename T>
class BoxTest : public testing::Test {};

template <typename Scalar_, int Dim_>
class BoxTest<Box<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest() : box(kDynamicBoxDim) {}

  Box<Scalar_, Dim_> box;
};

template <typename Scalar_, int Dim_>
class BoxTest<BoxMap<Scalar_, Dim_>> : public testing::Test {
 public:
  BoxTest()
      : min(Dim(Dim_)),
        max(Dim(Dim_)),
        box(min.data(), max.data(), Dim(Dim_)) {}

  std::vector<Scalar_> min;
  std::vector<Scalar_> max;
  BoxMap<Scalar_, Dim_> box;
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
  int dim = TypeParam::Dim;
  if (dim != kDynamicDim) {
    EXPECT_EQ(this->box.size(), dim);
  } else {
    EXPECT_EQ(this->box.size(), kDynamicBoxDim);
  }
}

TYPED_TEST(BoxTest, Accessors) {
  this->box.FillInverseMax();
  for (std::size_t i = 0; i < this->box.size(); ++i) {
    EXPECT_FLOAT_EQ(this->box.min(i), this->box.min()[i]);
    EXPECT_FLOAT_EQ(this->box.max(i), this->box.max()[i]);
  }
}

TYPED_TEST(BoxTest, FillInverseMax) {
  using Scalar = typename TypeParam::ScalarType;

  this->box.FillInverseMax();
  for (std::size_t i = 0; i < this->box.size(); ++i) {
    EXPECT_FLOAT_EQ(this->box.min(i), std::numeric_limits<Scalar>::max());
    EXPECT_FLOAT_EQ(this->box.max(i), std::numeric_limits<Scalar>::lowest());
  }
}

TYPED_TEST(BoxTest, Fit) {
  using Scalar = typename TypeParam::ScalarType;

  this->box.FillInverseMax();
  int dim = Dim(TypeParam::Dim);

  // Fit a point.
  std::vector<Scalar> p(static_cast<std::size_t>(dim), Scalar(0));
  this->box.Fit(p.data());
  for (std::size_t i = 0; i < this->box.size(); ++i) {
    EXPECT_FLOAT_EQ(this->box.min(i), Scalar(0));
    EXPECT_FLOAT_EQ(this->box.max(i), Scalar(0));
  }

  // Fit a box.
  std::vector<Scalar> min(static_cast<std::size_t>(dim), Scalar(-1));
  std::vector<Scalar> max(static_cast<std::size_t>(dim), Scalar(+1));
  this->box.Fit(BoxMap<Scalar, kDynamicDim>(min.data(), max.data(), dim));
  for (std::size_t i = 0; i < this->box.size(); ++i) {
    EXPECT_FLOAT_EQ(this->box.min(i), min[i]);
    EXPECT_FLOAT_EQ(this->box.max(i), max[i]);
  }
}
