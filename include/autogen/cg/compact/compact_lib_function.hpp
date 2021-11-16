#pragma once

#include "autogen/core/base.hpp"

namespace autogen {
struct CompactFunctionMetaData {
  int output_dim;
  int local_input_dim;
  int global_input_dim;
  int accumulated_output;
};

struct AbstractLibFunction {
  // loads the function pointer from a dynamic library handle
  template <typename FunctionPtrT>
  static FunctionPtrT load_function(const std::string &function_name,
                                    void *lib_handle) {
#if CPPAD_CG_SYSTEM_WIN
    auto ptr = (FunctionPtrT)GetProcAddress((HMODULE)lib_handle,
                                            function_name.c_str());
    if (!ptr) {
      throw std::runtime_error("Cannot load symbol '" + function_name +
                               "': error code " +
                               std::to_string(GetLastError()));
    }
#else
    auto ptr = (FunctionPtrT)dlsym(lib_handle, function_name.c_str());
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Cannot load symbol '" + function_name +
                               "': " + std::string(dlsym_error));
    }
#endif
    return ptr;
  }

  virtual ~AbstractLibFunction() = default;

  virtual void allocate(int num_total_threads) const = 0;
  virtual void deallocate() const = 0;
};

// type definitions for the function signatures
template <typename ScalarT>
struct LibFunctionTypes {
  using Scalar = ScalarT;
  // signature: (num_threads, output_vector)
  using LaunchFunctionPtrT = void (*)(int, ScalarT *);
  using SendLocalFunctionPtrT = bool (*)(int, const ScalarT *);
  using SendGlobalFunctionPtrT = bool (*)(const ScalarT *);
  using MetaDataFunctionPtrT = CompactFunctionMetaData (*)();
  using AllocateFunctionPtrT = void (*)(int);
  using DeallocateFunctionPtrT = void (*)();
};

template <typename FunctionTypes>
class CompactLibFunctionT : public AbstractLibFunction {
 public:
  template <class LibFunction>
  friend class CompactLibrary;

  std::string function_name;

  using Scalar = typename FunctionTypes::Scalar;
  using LaunchFunctionPtrT = typename FunctionTypes::LaunchFunctionPtrT;
  using SendLocalFunctionPtrT = typename FunctionTypes::SendLocalFunctionPtrT;
  using SendGlobalFunctionPtrT = typename FunctionTypes::SendGlobalFunctionPtrT;
  using MetaDataFunctionPtrT = typename FunctionTypes::MetaDataFunctionPtrT;
  using AllocateFunctionPtrT = typename FunctionTypes::AllocateFunctionPtrT;
  using DeallocateFunctionPtrT = typename FunctionTypes::DeallocateFunctionPtrT;

 protected:
  CompactFunctionMetaData meta_data_{};
  bool is_available_{false};

  // remembers for how many threads the last allocation took place
  mutable int last_allocation_threads_{0};

 public:
  bool is_available() const { return is_available_; };

  /**
   * Global input dimension.
   */
  int global_input_dim() const { return meta_data_.global_input_dim; }
  /**
   * Input dimension per thread.
   */
  int local_input_dim() const { return meta_data_.local_input_dim; }
  /**
   * Output dimension per thread.
   */
  int output_dim() const { return meta_data_.output_dim; }
  /**
   * Determines whether the output is accumulated over all threads.
   */
  bool accumulated_output() const { return meta_data_.accumulated_output; }

  CompactLibFunctionT() = default;
  virtual ~CompactLibFunctionT() {
    if (!is_available_ && last_allocation_threads_ != 0) {
      // function is not available but memory has been allocated
      assert(false);
    }
    if (is_available_) {
      deallocate();
    }
  }

  virtual void load(const std::string &function_name, void *lib_handle) {
    this->function_name = function_name;
    try {
      fun_ = load_function<LaunchFunctionPtrT>(function_name, lib_handle);
      is_available_ = true;
    } catch (const std::runtime_error &) {
      is_available_ = false;
    }
    if (is_available_) {
      auto meta_data_fun = load_function<MetaDataFunctionPtrT>(
          function_name + "_meta", lib_handle);
      meta_data_ = meta_data_fun();
      allocate_ = load_function<AllocateFunctionPtrT>(
          function_name + "_allocate", lib_handle);
      deallocate_ = load_function<DeallocateFunctionPtrT>(
          function_name + "_deallocate", lib_handle);
      try {
        send_global_fun_ = load_function<SendGlobalFunctionPtrT>(
            function_name + "_send_global", lib_handle);
      } catch (...) {
        // we ignore the case when there is no global send function
      }
      send_local_fun_ = load_function<SendLocalFunctionPtrT>(
          function_name + "_send_local", lib_handle);
    }
  }

  virtual void allocate(int num_total_threads) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    if (last_allocation_threads_ == num_total_threads) {
      // already allocated
      return;
    } else if (last_allocation_threads_ > 0) {
      std::cerr << "Memory for function \"" << function_name
                << "\" has been previously allocated for "
                << last_allocation_threads_
                << " thread(s). Automatically deallocating memory first...\n";
      deallocate();
    }
    allocate_(num_total_threads);
    last_allocation_threads_ = num_total_threads;
  }

  virtual void deallocate() const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    if (last_allocation_threads_ == 0) {
      return;
    }
    deallocate_();
    last_allocation_threads_ = 0;
  }

  virtual bool operator()(int num_total_threads, Scalar *output,
                          const Scalar *input) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    allocate(num_total_threads);
    assert(fun_);
    bool status = true;
    if (send_global_fun_) {
      status = send_global_input(input);
      assert(status);
    }
    if (!status) {
      return false;
    }
    status = send_local_input(num_total_threads,
                              &(input[meta_data_.global_input_dim]));
    assert(status);
    if (!status) {
      return false;
    }

    return execute_(num_total_threads, output);
  }

  virtual bool operator()(const std::vector<Scalar> &input,
                          std::vector<Scalar> &output) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    return (*this)(1, (Scalar*) &output[0], (const Scalar*) &input[0]);
  }

  virtual bool operator()(std::vector<std::vector<Scalar>> *thread_outputs,
                          const std::vector<std::vector<Scalar>> &local_inputs,
                          const std::vector<Scalar> &global_input = {}) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }

    allocate(static_cast<int>(local_inputs.size()));

    assert(fun_);
    bool status = true;
    status = send_local_input(local_inputs);
    assert(status);
    if (!status) {
      return false;
    }
    if (!global_input.empty()) {
      status = send_global_input(global_input);
      assert(status);
      if (!status) {
        return false;
      }
    }

    int num_total_threads = static_cast<int>(local_inputs.size());
    Scalar *output = new Scalar[num_total_threads * meta_data_.output_dim];

    status &= execute_(num_total_threads, output);

    // assign thread-wise outputs
    std::size_t i = 0;
    if (meta_data_.accumulated_output) {
      thread_outputs->resize(1);
      (*thread_outputs)[0].resize(output_dim());
      for (; i < output_dim(); ++i) {
        (*thread_outputs)[0][i] = output[i];
      }
    } else {
      thread_outputs->resize(local_inputs.size());
      for (auto &thread : *thread_outputs) {
        thread.resize(output_dim());
        for (Scalar &t : thread) {
          t = output[i];
          ++i;
        }
      }
    }

    delete[] output;
    return status;
  }

  virtual bool send_local_input(int num_total_threads,
                                const Scalar *input) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    assert(send_local_fun_);
    return send_local_fun_(num_total_threads, input);
  }
  virtual bool send_local_input(
      const std::vector<std::vector<Scalar>> &thread_inputs) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    assert(send_local_fun_);
    if (thread_inputs.empty() || static_cast<int>(thread_inputs[0].size()) !=
                                     meta_data_.local_input_dim) {
      assert(false);
      return false;
    }
    auto num_total_threads = static_cast<int>(thread_inputs.size());
    Scalar *input = new Scalar[thread_inputs[0].size() * num_total_threads];
    std::size_t i = 0;
    for (const auto &thread : thread_inputs) {
      for (const Scalar &t : thread) {
        input[i] = t;
        ++i;
      }
    }
    bool status = send_local_input(num_total_threads, input);
    delete[] input;
    return status;
  }
  virtual bool send_local_input(
      const std::vector<Scalar> &thread_inputs) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    assert(send_local_fun_);
    auto num_total_threads =
        static_cast<int>(thread_inputs.size() / meta_data_.local_input_dim);
    return send_local_fun_(num_total_threads, thread_inputs.data());
  }

  virtual bool send_global_input(const Scalar *input) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    assert(send_global_fun_);
    return send_global_fun_(input);
  }

  virtual bool send_global_input(const std::vector<Scalar> &input) const {
    if (!is_available_) {
      throw std::runtime_error("Library function \"" + function_name +
                               "\" is not available.");
    }
    assert(send_global_fun_);
    if (static_cast<int>(input.size()) != meta_data_.global_input_dim) {
      assert(false);
      return false;
    }
    return send_global_input(input.data());
  }

 protected:
  LaunchFunctionPtrT fun_{nullptr};
  AllocateFunctionPtrT allocate_{nullptr};
  DeallocateFunctionPtrT deallocate_{nullptr};
  SendGlobalFunctionPtrT send_global_fun_{nullptr};
  SendLocalFunctionPtrT send_local_fun_{nullptr};

  virtual bool execute_(int num_threads, Scalar *output) const {
    if constexpr (std::is_same_v<
                      LaunchFunctionPtrT,
                      typename LibFunctionTypes<Scalar>::LaunchFunctionPtrT>) {
      fun_(num_threads, output);
    } else {
      throw std::runtime_error(
          "LibFunction::execute_(num_threads, output) needs to be implemented "
          "since the type of the launch function pointer differs from the "
          "default CompactLibFunction launch function pointer type.");
      // static_assert(
      //     false,
      //     "LibFunction::execute_(num_threads, output) needs to be implemented
      //     " "since the type of the launch function pointer differs from the "
      //     "default CompactLibFunction launch function pointer type.");
    }
    return true;
  }
};

typedef CompactLibFunctionT<LibFunctionTypes<BaseScalar>> CompactLibFunction;
}  // namespace autogen