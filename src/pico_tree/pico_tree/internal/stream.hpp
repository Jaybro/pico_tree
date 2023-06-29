#pragma once

#include <fstream>
#include <vector>

namespace pico_tree {

namespace internal {

//! \brief Returns an std::fstream given a filename.
//! \details Convenience function that throws an std::runtime_error in case it
//! is unable to open the stream.
inline std::fstream OpenStream(
    std::string const& filename, std::ios_base::openmode mode) {
  std::fstream stream(filename, mode);

  if (!stream.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  return stream;
}

//! \brief The Stream class is an std::iostream wrapper that helps read and
//! write various simple data types.
class Stream {
 public:
  //! \brief Constructs a Stream using an input std::iostream.
  Stream(std::iostream& stream) : stream_(stream) {}

  //! \brief Reads a single value from the stream.
  //! \tparam T Type of the value.
  template <typename T>
  inline void Read(T& value) {
    stream_.read(reinterpret_cast<char*>(&value), sizeof(T));
  }

  //! \brief Reads a vector of values from the stream.
  //! \details Reads the size of the vector followed by all its elements.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Read(std::vector<T>& values) {
    typename std::vector<T>::size_type size;
    Read(size);
    values.resize(size);
    Read(size, values.data());
  }

  //! \brief Reads an array of values from the stream.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Read(std::size_t size, T* values) {
    stream_.read(reinterpret_cast<char*>(values), sizeof(T) * size);
  }

  //! \brief Writes a single value to the stream.
  //! \tparam T Type of the value.
  template <typename T>
  inline void Write(T const& value) {
    stream_.write(reinterpret_cast<char const*>(&value), sizeof(T));
  }

  //! \brief Writes a vector of values to the stream.
  //! \details Writes the size of the vector followed by all its elements.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Write(std::vector<T> const& values) {
    Write(values.size());
    stream_.write(
        reinterpret_cast<char const*>(&values[0]), sizeof(T) * values.size());
  }

  //! \brief Writes an array of values to the stream.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Write(T const* values, std::size_t size) {
    stream_.write(reinterpret_cast<char const*>(values), sizeof(T) * size);
  }

 private:
  //! \brief Wrapped stream.
  std::iostream& stream_;
};

}  // namespace internal

}  // namespace pico_tree
