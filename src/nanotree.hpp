#pragma once

#include <cassert>
#include <cmath>
#include <numeric>
#include <stack>
#include <vector>

// TODO(remove)
#include <glog/logging.h>

namespace nanotree {

//! Value used for any template Dims parameter to determine the dimensions
//! compile time.
constexpr int kRuntimeDims = -1;

namespace internal {

template <typename Derived>
class Traits;

template <typename T>
class ItemBuffer {
 public:
  ItemBuffer(std::size_t size) { buffer_.reserve(size); }

  template <typename... Args>
  inline T* MakeItem(Args&&... args) {
    buffer_.emplace_back(std::forward<Args>(args)...);
    return &buffer_.back();
  }

 private:
  std::vector<T> buffer_;
};

template <typename T>
inline T Sum(T const a) {
  return (a == 1) ? 1 : a + Sum(a - 1);
}

inline std::size_t Depth(std::size_t num_points, std::size_t ignored) {
  return std::round(std::log2(static_cast<double>(num_points))) - ignored;
}

inline std::size_t MaxBranchesFromPoints(std::size_t num_points) {
  return num_points - 1;
}

inline std::size_t MaxNodesFromPoints(std::size_t num_points) {
  return num_points * 2 - 1;
}

inline std::size_t MaxNodesFromDepth(std::size_t depth) { return Sum(depth); }

//! \private  Returns a permutation based on the sorted input.
template <typename Index, typename Compare>
std::vector<Index> SortPermutation(Index num_points, Compare compare) {
  std::vector<Index> p(num_points);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), std::move(compare));
  return p;
}

template <int Dim>
struct Dimensions {
  inline static constexpr int Get(int dim) { return Dim; }
};

template <>
struct Dimensions<kRuntimeDims> {
  inline static int Get(int dim) { return dim; }
};

template <typename Index, typename Scalar, int Dim, typename Points>
class RangeLayer {
 public:
  struct Item {
    inline std::pair<Index, Index> range() const {
      return std::make_pair(left, right);
    }

    Index index;
    Index left;
    Index right;
  };

  RangeLayer(Points const& points, Index num_points)
      : points_{points},
        dim_(points.num_dimensions() - 1),
        items_{static_cast<std::size_t>(num_points)} {}

  inline Index LowerBound(Scalar value) const {
    return std::lower_bound(
               items_.cbegin(),
               items_.cend(),
               value,
               [this](Item const& a, Scalar const b) {
                 return operator()(a) < b;
               }) -
           items_.cbegin();
  }

  inline Index UpperBound(Index lower_bound, Scalar value) const {
    return std::upper_bound(
               items_.cbegin() + lower_bound,
               items_.cend(),
               value,
               [this](Scalar const a, Item const& b) {
                 return a < operator()(b);
               }) -
           items_.cbegin();
  }

  inline std::vector<Item> const& data() const { return items_; }
  inline std::vector<Item>& data() { return items_; }
  inline std::pair<Index, Index> range(Index i) const {
    return items_[i].range();
  }

 private:
  //! Returns the value of a point having sorted index \p i .
  inline Scalar operator()(Item i) const {
    return points_(i.index, Dimensions<Dim>::Get(dim_));
  }

  Points const& points_;
  int dim_;
  std::vector<Item> items_;
};

template <typename Derived, typename Index, typename Scalar>
struct Node {
  union Data {
    struct Branch {
      Scalar split;
    };

    struct Leaf {
      // Probably make this begin and end index
      Index index;
    };

    Branch branch;
    Leaf leaf;
  };

  inline bool IsBranch() const { return left != nullptr && right != nullptr; }

  Data data;
  Derived* left;
  Derived* right;
};

}  // namespace internal

template <typename Index, typename Scalar>
class IPoints {
 public:
  //! Returns dimension \p dimension of point \p index:
  virtual inline Scalar operator()(
      Index const index, Index const dimension) const = 0;

  //! Returns the amount of spatial dimensions of the points.
  virtual inline Index num_dimensions() const = 0;

  //! Returns the number of points.
  virtual inline Index num_points() const = 0;
};

template <typename Index, typename Scalar, typename Points>
class RangeTree1d {
 public:
  RangeTree1d(Points const& points)
      : points_{points},
        sorted_{internal::SortPermutation(
            points_.num_points(), [this](int i, int j) -> bool {
              return operator()(i) < operator()(j);
            })} {
    assert(points_.num_points() > 0);
  }

  //! Back-inserts the indices into \p indices representing all points
  //! within the range [ \p min ... \p max ].
  inline void SearchRangeNd(
      Scalar min, Scalar max, std::vector<Index>* indices) const {
    auto it_lower = std::lower_bound(
        sorted_.cbegin(), sorted_.cend(), min, [this](Index a, Scalar b) {
          return operator()(a) < b;
        });

    auto it_upper = std::upper_bound(
        it_lower, sorted_.cend(), max, [this](Scalar a, Index b) {
          return a < operator()(b);
        });

    std::copy(it_lower, it_upper, std::back_inserter(*indices));
  }

  //! Returns the index of a point closest to \p value .
  inline Index SearchNearestNd(Scalar const& value) const {
    auto it = std::lower_bound(
        sorted_.cbegin(), sorted_.cend(), value, [this](Index a, Scalar b) {
          return operator()(a) < b;
        });

    if (it != sorted_.begin()) {
      auto prev = std::prev(it);
      if ((operator()(*it) - value) > (value - operator()(*prev))) {
        return *prev;
      }
    }

    return *it;
  }

 private:
  //! Returns the value of a point having sorted index \p i .
  inline Scalar operator()(Index i) const { return points_(i, 0); }

  Points const& points_;
  std::vector<Index> sorted_;
};

template <typename Index, typename Scalar, typename Points>
class RangeTree2d {
 private:
  using Layer = internal::RangeLayer<Index, Scalar, 1, Points>;
  using Item = typename Layer::Item;

  struct Node2d : public internal::Node<Node2d, Index, Scalar> {
    Layer* layer;
  };

  using Node = Node2d;

 public:
  RangeTree2d(Points const& points)
      : points_{points},
        nodes_{internal::MaxNodesFromPoints(points_.num_points())},
        layers_{internal::MaxNodesFromPoints(points_.num_points())},
        root_{MakeTree()} {
    assert(points_.num_points() > 0);
  }

  // TODO Generalize interface
  //! Perform range search in O(log_2 n) time.
  inline void SearchRangeNd(
      Scalar const min_x,
      Scalar const min_y,
      Scalar const max_x,
      Scalar const max_y,
      std::vector<Index>* indices) const {
    assert(min_x < max_x);
    assert(min_y < max_y);
    // Find node where min_x and max_x split.
    Node* split = root_;
    if (split->IsBranch()) {
      if (max_x < split->data.branch.split) {
        do {
          // std::cout << "split: " << split->data.branch.split << std::endl;
          split = split->left;
        } while (split->IsBranch() && max_x < split->data.branch.split);
      } else if (min_x > split->data.branch.split) {
        do {
          // std::cout << "split: " << split->data.branch.split << std::endl;
          split = split->right;
        } while (split->IsBranch() && min_x > split->data.branch.split);
      }

      // std::cout << "split: " << split->data.branch.split << std::endl;

      if (split->IsBranch()) {
        auto const lower_bound = split->layer->LowerBound(min_y);
        std::pair<Index, Index> c_lower = split->layer->range(lower_bound);
        std::pair<Index, Index> c_upper =
            split->layer->range(split->layer->UpperBound(lower_bound, max_y));
        // We follow the left track to the bottom.
        Node* track = split->left;
        std::pair<Index, Index> track_c_lower =
            track->layer->range(c_lower.first);
        std::pair<Index, Index> track_c_upper =
            track->layer->range(c_upper.first);

        while (track->IsBranch()) {
          if (min_x <= track->data.branch.split) {
            // std::cout << "left give: " << track->data.branch.split <<
            // std::endl;
            ReportIndices(
                track->right->layer->data(),
                track_c_lower.second,
                track_c_upper.second,
                indices);
            track = track->left;
            track_c_lower = track->layer->range(track_c_lower.first);
            track_c_upper = track->layer->range(track_c_upper.first);
          } else {
            // std::cout << "left move: " << track->data.branch.split <<
            // std::endl;
            track = track->right;
            track_c_lower = track->layer->range(track_c_lower.second);
            track_c_upper = track->layer->range(track_c_upper.second);
          }
        }
        // Last left leaf
        if (min_x <= points_(track->data.leaf.index, 0) &&
            min_y <= points_(track->data.leaf.index, 1)) {
          // std::cout << "left leaf: " << track->data.leaf.index << std::endl;
          indices->push_back(track->data.leaf.index);
        }

        // We follow the right track to the bottom.
        track = split->right;
        track_c_lower = track->layer->range(c_lower.second);
        track_c_upper = track->layer->range(c_upper.second);

        while (track->IsBranch()) {
          if (max_x >= track->data.branch.split) {
            // std::cout << "right give: " << track->data.branch.split <<
            // std::endl;
            ReportIndices(
                track->left->layer->data(),
                track_c_lower.first,
                track_c_upper.first,
                indices);
            track = track->right;
            track_c_lower = track->layer->range(track_c_lower.second);
            track_c_upper = track->layer->range(track_c_upper.second);
          } else {
            // std::cout << "right move: " << track->data.branch.split <<
            // std::endl;
            track = track->left;
            track_c_lower = track->layer->range(track_c_lower.first);
            track_c_upper = track->layer->range(track_c_upper.first);
          }
        }

        // Last right leaf
        if (max_x >= points_(track->data.leaf.index, 0) &&
            max_y >= points_(track->data.leaf.index, 1)) {
          // std::cout << "right leaf: " << left->data.leaf.index <<
          // std::endl;
          indices->push_back(track->data.leaf.index);
        }
      } else {
        // We never found a split node and ended up in a leaf.
        if (min_x <= points_(split->data.leaf.index, 0) &&
            min_y <= points_(split->data.leaf.index, 1) &&
            max_x >= points_(split->data.leaf.index, 0) &&
            max_y >= points_(split->data.leaf.index, 1)) {
          indices->push_back(split->data.leaf.index);
        }
      }
    } else {
      // The root is a leaf.
      if (min_x <= points_(split->data.leaf.index, 0) &&
          min_y <= points_(split->data.leaf.index, 1) &&
          max_x >= points_(split->data.leaf.index, 0) &&
          max_y >= points_(split->data.leaf.index, 1)) {
        indices->push_back(split->data.leaf.index);
      }
    }
  }

 private:
  inline Scalar operator()(Index i, Index d) const { return points_(i, d); }

  //! Builds the tree in O(3 * n log n) time.
  inline Node* MakeTree() {
    auto const& points = points_;

    // Sorted x indices that directly refer to a point.
    std::vector<Index> direct_p_by_x{internal::SortPermutation(
        points.num_points(), [this](Index i, Index j) -> bool {
          return operator()(i, 0) < operator()(j, 0);
        })};

    // Sorted y indices that link to the points via the x dimension.
    std::vector<Index> linked_x_by_y{internal::SortPermutation(
        points.num_points(), [this, &direct_p_by_x](Index i, Index j) -> bool {
          return operator()(direct_p_by_x[i], 1) < operator()(
                                                       direct_p_by_x[j], 1);
        })};

    // Root associate layer.
    Layer* parent = layers_.MakeItem(points_, points.num_points());
    for (Index i = 0; i < points.num_points(); ++i) {
      parent->data()[i].index = direct_p_by_x[linked_x_by_y[i]];
    }

    std::vector<Index> buffer(linked_x_by_y.size());
    return SplitIndices(
        direct_p_by_x,
        direct_p_by_x.size() / 2,
        parent,
        &linked_x_by_y,
        &buffer);
  }

  inline Node* SplitIndices(
      std::vector<Index> const& direct_p_by_x,
      Index const split,
      Layer* parent,
      std::vector<Index>* p_front,
      std::vector<Index>* p_back) {
    // Leaf
    if (parent->data().size() == 1) {
      Node* node = nodes_.MakeItem();
      node->data.leaf.index = direct_p_by_x[split];
      node->layer = parent;
      node->left = nullptr;
      node->right = nullptr;
      return node;
    }

    std::vector<Index> const& front = *p_front;
    std::vector<Index>& back = *p_back;

    // Right may be one bigger than left
    Index const parent_size = parent->data().size();
    Index const left_size = parent_size / 2;
    Index const right_size = parent_size - left_size;
    Index const left_offset = split - left_size;
    Index const right_offset = split;
    Layer* left = layers_.MakeItem(points_, left_size);
    Layer* right = layers_.MakeItem(points_, right_size);
    Index left_count = 0;
    Index right_count = 0;

    // Cascade to the next layer.
    for (Index i = 0; i < parent_size; ++i) {
      // Index in the next layer to the smallest value which is greater or
      // equal to the current one. Since the sequence is ordened, the current
      // value to be inserted must be the smallest one. The index will be the
      // next to be inserted.
      // TODO Test this later =D
      // NOTE: In case there is an equals range, we should check the last
      // existing entry for equality. Practically, this gains us nothing as we
      // should always arrive at the start of the equals range (or the end).
      parent->data()[i].left = left_count;
      parent->data()[i].right = right_count;
      // We move the sorted y values into the left or right node based on the
      // sorted x indices while keeping their relative ordering.
      // Split is the first index on the right side.
      Index const index_prev_dim = front[left_offset + i];
      if (index_prev_dim < right_offset) {
        left->data()[left_count].index = parent->data()[i].index;
        back[left_offset + left_count] = index_prev_dim;
        ++left_count;
      } else {
        right->data()[right_count].index = parent->data()[i].index;
        back[right_offset + right_count] = index_prev_dim;
        ++right_count;
      }
    }

    Node* node = nodes_.MakeItem();
    node->data.branch.split = operator()(direct_p_by_x[split], 0);
    node->layer = parent;
    node->left = SplitIndices(
        direct_p_by_x, left_offset + left_size / 2, left, p_back, p_front);
    node->right = SplitIndices(
        direct_p_by_x, right_offset + right_size / 2, right, p_back, p_front);
    return node;
  }

  inline void ReportIndices(
      std::vector<Item> const& items,
      Index const lower_bound,
      Index const upper_bound,
      std::vector<Index>* indices) const {
    // For MingW GCC 9.2.0, this was hands down the fastest for
    // copying indices. Vs. for_each, or any other type of for/while loop.
    std::transform(
        items.cbegin() + lower_bound,
        items.cbegin() + upper_bound,
        std::back_inserter(*indices),
        [](Item const& i) { return i.index; });
  }

  Points const& points_;
  internal::ItemBuffer<Node> nodes_;
  internal::ItemBuffer<Layer> layers_;
  Node* root_;
};

}  // namespace nanotree