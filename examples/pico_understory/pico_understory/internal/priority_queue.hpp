#pragma once

#include <vector>

#include "pico_tree/internal/search_visitor.hpp"

namespace pico_tree::internal {

template <typename P_, typename T_>
class PriorityQueue {
 public:
  inline explicit PriorityQueue() = default;

  inline std::pair<P_, T_>& Front() { return buffer_.front(); }

  inline std::pair<P_, T_> const& Front() const { return buffer_.front(); }

  inline void PopFront() {
    // TODO For small queues it's probably not worth it to use a deque.
    buffer_.erase(buffer_.begin());
  }

  inline void PushBack(P_ priority, T_ item) {
    if (buffer_.size() < buffer_.capacity()) {
      if (buffer_.empty()) {
        buffer_.push_back({priority, item});
      } else if (buffer_.back().first < priority) {
        buffer_.push_back({priority, item});
      } else {
        buffer_.push_back(buffer_.back());
        InsertSorted(
            buffer_.begin(), std::prev(buffer_.end()), {priority, item});
      }
    } else if (buffer_.back().first > priority) {
      InsertSorted(buffer_.begin(), buffer_.end(), {priority, item});
    }
  }

  void reserve(std::size_t capacity) { buffer_.reserve(capacity); }

  std::size_t capacity() const { return buffer_.capacity(); }

  std::size_t size() const { return buffer_.size(); }

  bool empty() const { return buffer_.empty(); }

 private:
  std::vector<std::pair<P_, T_>> buffer_;
};

}  // namespace pico_tree::internal
