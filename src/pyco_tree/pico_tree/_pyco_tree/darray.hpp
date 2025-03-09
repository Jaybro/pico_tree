#pragma once

#include <pybind11/numpy.h>

#include <memory>
#include <vector>

#include "def_core.hpp"

namespace pyco_tree {

namespace internal {

class darray_impl_base {
 public:
  virtual ~darray_impl_base() = default;

  virtual pybind11::array operator[](std::size_t i) = 0;

  virtual std::unique_ptr<darray_impl_base> copy(
      std::size_t begin, std::size_t step, std::size_t count) const = 0;

  virtual std::size_t size() const = 0;

  virtual bool empty() const = 0;

  virtual pybind11::dtype dtype() const = 0;
};

template <typename T_>
class darray_impl : public darray_impl_base {
 public:
  static_assert(
      std::is_standard_layout_v<T_> && std::is_trivial_v<T_>,
      "Type T_ doesn't have a standard layout or isn't trivial.");

  darray_impl(std::vector<std::vector<T_>> array) : array_(std::move(array)) {}

  pybind11::array operator[](std::size_t i) override {
    // A NumPy array has ownership of its own data when it is created
    // with default arguments. This causes it to copy the data from the
    // input pointer unless we tell it that an other object should own
    // its data. In this case that other object equals py::none().
    // TODO It would have been nice if that could be this/self?
    // It is important that at the binding side of things we ensure that the
    // array is kept alive while the view is alive.
    // NOTE: At the time of writing an undocumented feature.

    // In case the size of a vector equals 0, its data pointer can equal
    // nullptr. When this happens, the library interface of numpy (as wrapped by
    // pybind11) will allocate some memory and store the address to that memory
    // instead of storing the nullptr address of the vector. This means that
    // each time we create a view for the same empty vector, the memory address
    // it stores may randomly change. This is not an issue, but good to document
    // here. See:
    //  * PyArray_NewFromDescr(...)
    //  * https://numpy.org/doc/1.13/reference/c-api.array.html
    return pybind11::array_t<T_, 0>(
        static_cast<py::ssize_t>(array_[i].size()),
        array_[i].data(),
        pybind11::none());
  }

  std::unique_ptr<darray_impl_base> copy(
      std::size_t begin, std::size_t step, std::size_t count) const override {
    std::vector<std::vector<T_>> array;
    array.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
      array.push_back(array_[begin]);
      begin += step;
    }

    return std::make_unique<darray_impl>(std::move(array));
  }

  std::size_t size() const override { return array_.size(); }

  bool empty() const override { return array_.empty(); }

  pybind11::dtype dtype() const override { return pybind11::dtype::of<T_>(); }

  std::vector<std::vector<T_>> const& data() const { return array_; }

  std::vector<std::vector<T_>>& data() { return array_; }

 private:
  std::vector<std::vector<T_>> array_;
};

}  // namespace internal

class darray {
  template <typename T_>
  //! \brief Wraps a type T_ and exposes it as a pointer to type T_.
  //! \details In some cases we are dependent on having a pointer interface for
  //! a variable that would otherwise go out of scope.
  class pointer_interface {
   public:
    inline pointer_interface(T_ array) : array_(std::move(array)) {}

    inline T_* operator->() { return &array_; }
    inline T_& operator*() & { return array_; }
    inline T_&& operator*() && { return std::move(array_); }

    inline operator T_*() { return &array_; }

   private:
    T_ array_;
  };

  //! \brief The darray_iterator class allows iterating over the contained
  //! vectors and presenting them as numpy ndarray views.
  class darray_iterator {
   public:
    // clang-format off
    using iterator_category = std::random_access_iterator_tag;
    using size_type         = std::size_t;
    using difference_type   = std::ptrdiff_t;
    using value_type        = pybind11::array;
    using pointer           = pointer_interface<pybind11::array>;
    using reference         = pybind11::array;
    // clang-format on

    //! \brief Constructs an Interator from an array and an index.
    darray_iterator(internal::darray_impl_base* array, difference_type index)
        : array_(array), index_(index) {}

    //! \private
    value_type operator[](difference_type const i) {
      return array_->operator[](static_cast<size_type>(index_ + i));
    }

    //! \private
    pointer operator->() {
      // A pointer interface.
      return pointer(array_->operator[](static_cast<size_type>(index_)));
    }

    //! \private
    reference operator*() {
      // A reference is just a copy.
      return array_->operator[](static_cast<size_type>(index_));
    }

    //! \private
    darray_iterator& operator++() {
      index_++;
      return *this;
    }

    //! \private
    darray_iterator operator++(int) {
      darray_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    //! \private
    darray_iterator& operator--() {
      index_--;
      return *this;
    }

    //! \private
    darray_iterator operator--(int) {
      darray_iterator tmp = *this;
      --(*this);
      return tmp;
    }

    //! \private
    friend bool operator==(darray_iterator const& a, darray_iterator const& b) {
      return a.index_ == b.index_ && a.array_ == b.array_;
    };

    //! \private
    friend bool operator!=(darray_iterator const& a, darray_iterator const& b) {
      return a.index_ != b.index_ || a.array_ != b.array_;
    };

   private:
    internal::darray_impl_base* array_;
    difference_type index_;
  };

 public:
  using difference_type = darray_iterator::difference_type;
  using size_type = darray_iterator::size_type;
  using value_type = darray_iterator::value_type;
  using pointer = darray_iterator::pointer;
  using reference = pybind11::array;

  darray() = default;

  template <typename T_>
  explicit darray(std::vector<std::vector<T_>> darray)
      : impl_(std::make_unique<internal::darray_impl<T_>>(std::move(darray))) {}

  darray(pybind11::dtype const dtype) {
    if (dtype.equal(pybind11::dtype::of<neighbor_f>())) {
      reset(std::vector<std::vector<neighbor_f>>());
    } else if (dtype.equal(pybind11::dtype::of<neighbor_d>())) {
      reset(std::vector<std::vector<neighbor_d>>());
    } else if (dtype.equal(pybind11::dtype::of<int>())) {
      reset(std::vector<std::vector<int>>());
    } else {
      throw std::invalid_argument("dtype not supported");
    }
  }

  template <typename T_>
  void reset(std::vector<std::vector<T_>> darray) {
    impl_.reset(new internal::darray_impl<T_>(std::move(darray)));
  }

  darray copy(std::size_t begin, std::size_t step, std::size_t count) const {
    return darray(impl_->copy(begin, step, count));
  }

  pybind11::array operator[](size_type const i) { return impl_->operator[](i); }

  size_type size() const { return impl_->size(); }

  bool empty() const { return impl_->empty(); }

  pybind11::dtype dtype() const { return impl_->dtype(); }

  template <typename T_>
  std::vector<std::vector<T_>> const& data() const {
    if (!impl_) {
      throw std::runtime_error("array is uninitialized");
    }

    internal::darray_impl<T_> const* ptr =
        dynamic_cast<internal::darray_impl<T_> const*>(impl_.get());

    if (ptr == nullptr) {
      throw std::runtime_error("incorrect data type requested");
    }

    return ptr->data();
  }

  template <typename T_>
  std::vector<std::vector<T_>>& data() {
    if (!impl_) {
      throw std::runtime_error("array is uninitialized");
    }

    internal::darray_impl<T_>* ptr =
        dynamic_cast<internal::darray_impl<T_>*>(impl_.get());

    if (ptr == nullptr) {
      throw std::runtime_error("incorrect data type requested");
    }

    return ptr->data();
  }

  template <typename T_>
  std::vector<std::vector<T_>> const& data_unchecked() const {
    return static_cast<internal::darray_impl<T_> const*>(impl_.get())->data();
  }

  template <typename T_>
  std::vector<std::vector<T_>>& data_unchecked() {
    return static_cast<internal::darray_impl<T_>*>(impl_.get())->data();
  }

  darray_iterator begin() {
    // Start at index 0
    return darray_iterator(impl_.get(), 0);
  }

  darray_iterator end() {
    // The end is 1 past the last item.
    return darray_iterator(impl_.get(), static_cast<difference_type>(size()));
  }

 private:
  darray(std::unique_ptr<internal::darray_impl_base> array)
      : impl_(std::move(array)) {}

  std::unique_ptr<internal::darray_impl_base> impl_;
};

}  // namespace pyco_tree
