#include <gtest/gtest.h>

#include <pico_tree/internal/segment.hpp>

namespace pti = pico_tree::internal;

namespace {

inline constexpr float abs_error = 0.00001f;

}

TEST(SegmentR1Test, Distance) {
  EXPECT_NEAR(
      pti::segment_r1<float>(0.6f, 0.9f).distance(0.7f), 0.0f, abs_error);
  EXPECT_NEAR(
      pti::segment_r1<float>(0.4f, 0.6f).distance(0.3f), 0.1f, abs_error);
}

TEST(SegmentS1Test, DistanceMinMax) {
  EXPECT_NEAR(
      pti::segment_s1<float>(0.4f, 0.6f).distance_min_max(0.3f),
      0.1f,
      abs_error);
  EXPECT_NEAR(
      pti::segment_s1<float>(0.4f, 0.6f).distance_min_max(0.5f),
      0.0f,
      abs_error);
  EXPECT_NEAR(
      pti::segment_s1<float>(0.9f, 1.0f).distance_min_max(0.1f),
      0.1f,
      abs_error);
}

TEST(SegmentS1Test, DistanceMaxMin) {
  EXPECT_NEAR(
      pti::segment_s1<float>(0.9f, 0.1f).distance_max_min(0.0f),
      0.0f,
      abs_error);
  EXPECT_NEAR(
      pti::segment_s1<float>(0.9f, 0.1f).distance_max_min(0.2f),
      0.1f,
      abs_error);
}

TEST(SegmentS1Test, Distance) {
  EXPECT_NEAR(
      pti::segment_s1<float>(0.9f, 0.1f).distance(0.2f), 0.1f, abs_error);
  EXPECT_NEAR(
      pti::segment_s1<float>(0.4f, 0.6f).distance(0.3f), 0.1f, abs_error);
}
