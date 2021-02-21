#pragma once

#include <chrono>
#include <iostream>
#include <string>

// A simple timer that will print its life time to the standard out.
class ScopedTimer {
 public:
  ScopedTimer(std::string const& name)
      : name_{name},
        start_{std::chrono::high_resolution_clock::now()},
        times_{1} {}

  ScopedTimer(std::string const& name, std::size_t times)
      : name_{name},
        start_{std::chrono::high_resolution_clock::now()},
        times_{times} {}

  ~ScopedTimer() {
    std::chrono::duration<double> elapsed_seconds =
        std::chrono::high_resolution_clock::now() - start_;
    std::cout << "[" << name_
              << "] Elapsed time: " << (elapsed_seconds.count() * 1000.0)
              << " ms\n";

    if (times_ > 1) {
      std::cout << "[" << name_ << "] Average time: "
                << ((elapsed_seconds.count() / static_cast<double>(times_)) *
                    1000.0)
                << " ms\n";
    }
  }

  ScopedTimer(ScopedTimer const&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  ScopedTimer& operator=(ScopedTimer const&) = delete;
  ScopedTimer& operator=(ScopedTimer&&) = delete;

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
  std::size_t times_;
};
