#include <algorithm>
#include <filesystem>
#include <iostream>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/format/format_mnist.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/kd_forest.hpp>

template <typename U, typename T, std::size_t N>
std::array<U, N> Cast(std::array<T, N> const& i) {
  std::array<U, N> c;
  std::transform(i.begin(), i.end(), c.begin(), [](T a) -> U {
    return static_cast<U>(a);
  });
  return c;
}

template <std::size_t N>
std::vector<std::array<float, N>> Cast(
    std::vector<std::array<std::byte, N>> const& i) {
  std::vector<std::array<float, N>> c;
  std::transform(
      i.begin(),
      i.end(),
      std::back_inserter(c),
      [](std::array<std::byte, N> const& a) -> std::array<float, N> {
        return Cast<float>(a);
      });
  return c;
}

int main() {
  using ImageByte = std::array<std::byte, 28 * 28>;
  using ImageFloat = std::array<float, 28 * 28>;

  std::string fn_images_train = "train-images.idx3-ubyte";
  std::string fn_images_test = "t10k-images.idx3-ubyte";
  std::string fn_mnist_nns_gt = "mnist_nns_gt.bin";

  if (!std::filesystem::exists(fn_images_train)) {
    std::cout << fn_images_train << " doesn't exist." << std::endl;
    return 0;
  }

  if (!std::filesystem::exists(fn_images_test)) {
    std::cout << fn_images_test << " doesn't exist." << std::endl;
    return 0;
  }

  std::vector<ImageFloat> images_train;
  {
    std::vector<ImageByte> images_train_u8;
    pico_tree::ReadMnistImages(fn_images_train, images_train_u8);
    images_train = Cast(images_train_u8);
  }

  std::vector<ImageFloat> images_test;
  {
    std::vector<ImageByte> images_test_u8;
    pico_tree::ReadMnistImages(fn_images_test, images_test_u8);
    images_test = Cast(images_test_u8);
  }

  std::size_t max_leaf_size_ex = 16;
  std::size_t max_leaf_size_rp = 128;
  // With 8 trees we can get a precision of around 79%.
  // With 16 trees we can get a precision of around 93%.
  // With 32 trees we can get a precision of around 98%.
  std::size_t forest_size = 8;
  std::size_t count = images_test.size();
  std::vector<pico_tree::Neighbor<int, float>> nns(count);

  if (!std::filesystem::exists(fn_mnist_nns_gt)) {
    auto kd_tree = [&images_train, &max_leaf_size_ex]() {
      ScopedTimer t0("kd_tree build");
      return pico_tree::KdTree<std::reference_wrapper<std::vector<ImageFloat>>>(
          images_train, max_leaf_size_ex);
    }();

    {
      ScopedTimer t1("kd_tree query");
      for (std::size_t i = 0; i < nns.size(); ++i) {
        kd_tree.SearchNn(images_test[i], nns[i]);
      }
    }

    std::cout << "Writing " << fn_mnist_nns_gt << "." << std::endl;
    pico_tree::WriteBin(fn_mnist_nns_gt, nns);
  } else {
    std::cout << "Reading " << fn_mnist_nns_gt << "." << std::endl;
    pico_tree::ReadBin(fn_mnist_nns_gt, nns);
  }

  std::size_t equal = 0;

  // Building the tree takes roughly forest_size times longer to do versus the
  // regular KdTree. However, even with 32 trees, queries are more than a single
  // order of magnitude faster.
  {
    auto rkd_tree = [&images_train, &max_leaf_size_rp, &forest_size]() {
      ScopedTimer t0("kd_forest build");
      return pico_tree::KdForest<
          std::reference_wrapper<std::vector<ImageFloat>>>(
          images_train, max_leaf_size_rp, forest_size);
    }();

    ScopedTimer t1("kd_forest query");
    pico_tree::Neighbor<int, float> nn;
    for (std::size_t i = 0; i < nns.size(); ++i) {
      rkd_tree.SearchNn(images_test[i], nn);

      if (nns[i].index == nn.index) {
        ++equal;
      }
    }
  }

  std::cout << "Precision: "
            << (static_cast<float>(equal) / static_cast<float>(count))
            << std::endl;

  return 0;
}
