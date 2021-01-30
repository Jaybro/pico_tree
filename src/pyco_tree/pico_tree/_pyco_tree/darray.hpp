#pragma once

#include <pybind11/numpy.h>

#include <memory>
#include <vector>

#include "def_core.hpp"

namespace pyco_tree {

namespace internal {

class DArrayImplBase {
 public:
  virtual ~DArrayImplBase() = default;

  virtual pybind11::array operator[](std::size_t i) = 0;

  virtual std::unique_ptr<DArrayImplBase> Copy(
      std::size_t begin, std::size_t step, std::size_t count) const = 0;

  virtual std::size_t size() const = 0;

  virtual bool empty() const = 0;

  virtual pybind11::dtype dtype() const = 0;
};

template <typename T>
class DArrayImpl : public DArrayImplBase {
 public:
  static_assert(
      std::is_standard_layout<T>::value && std::is_trivial<T>::value,
      "Type T doesn't have a standard layout or isn't trivial.");

  DArrayImpl(std::vector<std::vector<T>> array) : array_(std::move(array)) {}

  pybind11::array operator[](std::size_t const i) override {
    // A NumPy array has ownership of its own data when it is created
    // with default arguments. This causes it to copy the data from the
    // input pointer unless we tell it that an other object should own
    // its data. In this case that other object equals py::none().
    // TODO It would have been nice if that could be this/self?
    // It is important that at the binding side of things we ensure that the
    // array is kept alive while the view is alive.
    // NOTE: At the time of writing an undocumented feature.
    return pybind11::array_t<T, 0>(
        array_[i].size(), array_[i].data(), pybind11::none());
  }

  std::unique_ptr<DArrayImplBase> Copy(
      std::size_t begin, std::size_t step, std::size_t count) const override {
    std::vector<std::vector<T>> array;
    array.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
      array.push_back(array_[begin]);
      begin += step;
    }

    return std::unique_ptr<DArrayImplBase>(new DArrayImpl(std::move(array)));
  }

  std::size_t size() const override { return array_.size(); }

  bool empty() const override { return array_.empty(); }

  pybind11::dtype dtype() const override { return pybind11::dtype::of<T>(); }

  std::vector<std::vector<T>> const& data() const { return array_; }

  std::vector<std::vector<T>>& data() { return array_; }

 private:
  std::vector<std::vector<T>> array_;
};

}  // namespace internal

// TODO Class names.
class DArray {
 public:
  template <typename T>
  //! \brief Wraps a type T and exposes it as a poiner to type T.
  //! \details In some cases we are dependent on having a pointer interface for
  //! a variable that would otherwise go out of scope.
  class PointerInterface {
   public:
    inline PointerInterface(T array) : array_(std::move(array)) {}

    inline T* operator->() { return &array_; }
    inline T& operator*() & { return array_; }
    inline T&& operator*() && { return std::move(array_); }

    inline operator T*() { return &array_; }

   private:
    T array_;
  };

  //! \brief The DArray Iterator class allows iterating over the contained
  //! vectors and presenting them as numpy ndarray views.
  class Iterator {
   public:
    // clang-format off
    using iterator_category = std::random_access_iterator_tag;
    using size_type         = std::size_t;
    using difference_type   = std::ptrdiff_t;
    using value_type        = pybind11::array;
    using pointer           = PointerInterface<pybind11::array>;
    using reference         = pybind11::array;
    // clang-format on

    //! \brief Constructs an Interator from an array and an index.
    Iterator(internal::DArrayImplBase* array, difference_type index)
        : array_(array), index_(index) {}

    //! \private
    value_type operator[](difference_type const i) {
      return array_->operator[](i);
    }

    //! \private
    pointer operator->() {
      // A pointer interface.
      return pointer(array_->operator[](index_));
    }

    //! \private
    reference operator*() {
      // A reference is just a copy.
      return array_->operator[](index_);
    }

    //! \private
    Iterator& operator++() {
      index_++;
      return *this;
    }

    //! \private
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    //! \private
    Iterator& operator--() {
      index_--;
      return *this;
    }

    //! \private
    Iterator operator--(int) {
      Iterator tmp = *this;
      --(*this);
      return tmp;
    }

    //! \private
    friend bool operator==(Iterator const& a, Iterator const& b) {
      return a.index_ == b.index_ && a.array_ == b.array_;
    };

    //! \private
    friend bool operator!=(Iterator const& a, Iterator const& b) {
      return a.index_ != b.index_ || a.array_ != b.array_;
    };

   private:
    internal::DArrayImplBase* array_;
    difference_type index_;
  };

  using difference_type = Iterator::difference_type;
  using size_type = Iterator::size_type;
  using value_type = Iterator::value_type;
  using pointer = Iterator::pointer;
  using reference = pybind11::array;

  DArray() = default;

  template <typename T>
  explicit DArray(std::vector<std::vector<T>> darray)
      : impl_(std::unique_ptr<internal::DArrayImplBase>(
            new internal::DArrayImpl<T>(std::move(darray)))) {}

  DArray(pybind11::dtype const dtype) {
    if (dtype.equal(pybind11::dtype::of<Neighborf>())) {
      Reset(std::vector<std::vector<Neighborf>>());
    } else if (dtype.equal(pybind11::dtype::of<Neighbord>())) {
      Reset(std::vector<std::vector<Neighbord>>());
    } else if (dtype.equal(pybind11::dtype::of<int>())) {
      Reset(std::vector<std::vector<int>>());
    } else {
      throw std::invalid_argument("Type not supported.");
    }
  }

  template <typename T>
  void Reset(std::vector<std::vector<T>> darray) {
    impl_.reset(new internal::DArrayImpl<T>(std::move(darray)));
  }

  DArray Copy(std::size_t begin, std::size_t step, std::size_t count) const {
    return DArray(impl_->Copy(begin, step, count));
  }

  pybind11::array operator[](size_type const i) { return impl_->operator[](i); }

  size_type size() const { return impl_->size(); }

  bool empty() const { return impl_->empty(); }

  pybind11::dtype dtype() const { return impl_->dtype(); }

  template <typename T>
  std::vector<std::vector<T>> const& data() const {
    if (!impl_) {
      throw std::runtime_error("Array is uninitialized.");
    }

    internal::DArrayImpl<T> const* ptr =
        dynamic_cast<internal::DArrayImpl<T> const*>(impl_.get());

    if (ptr == nullptr) {
      throw std::runtime_error("Incorrect data type requested.");
    }

    return ptr->data();
  }

  template <typename T>
  std::vector<std::vector<T>>& data() {
    if (!impl_) {
      throw std::runtime_error("Array is uninitialized.");
    }

    internal::DArrayImpl<T>* ptr =
        dynamic_cast<internal::DArrayImpl<T>*>(impl_.get());

    if (ptr == nullptr) {
      throw std::runtime_error("Incorrect data type requested.");
    }

    return ptr->data();
  }

  template <typename T>
  std::vector<std::vector<T>> const& data_unchecked() const {
    return static_cast<internal::DArrayImpl<T> const*>(impl_.get())->data();
  }

  template <typename T>
  std::vector<std::vector<T>>& data_unchecked() {
    return static_cast<internal::DArrayImpl<T>*>(impl_.get())->data();
  }

  Iterator begin() {
    // Start at index 0
    return Iterator(impl_.get(), 0);
  }

  Iterator end() {
    // The end is 1 past the last item.
    return Iterator(impl_.get(), static_cast<difference_type>(size()));
  }

 private:
  DArray(std::unique_ptr<internal::DArrayImplBase> array)
      : impl_(std::move(array)) {}

  std::unique_ptr<internal::DArrayImplBase> impl_;
};

}  // namespace pyco_tree
