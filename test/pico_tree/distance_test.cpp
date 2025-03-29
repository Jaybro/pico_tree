#include <gtest/gtest.h>

#include <pico_tree/metric.hpp>

namespace pt = pico_tree;

namespace {

inline constexpr float abs_error = 0.00001f;

}

TEST(DistanceTest, R1Distance) {
  EXPECT_NEAR(pt::r1_distance(1.9f, 1.1f), 0.8f, abs_error);
  EXPECT_NEAR(pt::r1_distance(1.1f, 1.9f), 0.8f, abs_error);
}

TEST(DistanceTest, S1Distance) {
  EXPECT_NEAR(pt::s1_distance(0.4f, 0.6f), 0.2f, abs_error);
  EXPECT_NEAR(pt::s1_distance(0.6f, 0.4f), 0.2f, abs_error);
  EXPECT_NEAR(pt::s1_distance(0.1f, 0.9f), 0.2f, abs_error);
}

TEST(DistanceTest, Squared) {
  EXPECT_NEAR(pt::squared(0.8f), 0.64f, abs_error);
}

TEST(DistanceTest, SquaredR1Distance) {
  EXPECT_NEAR(pt::squared_r1_distance(1.9f, 1.1f), 0.64f, abs_error);
  EXPECT_NEAR(pt::squared_r1_distance(1.1f, 1.9f), 0.64f, abs_error);
}

TEST(DistanceTest, SquaredS1Distance) {
  EXPECT_NEAR(pt::squared_s1_distance(0.4f, 0.6f), 0.04f, abs_error);
  EXPECT_NEAR(pt::squared_s1_distance(0.6f, 0.4f), 0.04f, abs_error);
  EXPECT_NEAR(pt::squared_s1_distance(0.1f, 0.9f), 0.04f, abs_error);
}
