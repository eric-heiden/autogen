#pragma once

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>
#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif

// clang-format off
#include "cuda/cuda_codegen.hpp"
#include "cuda/cuda_library_processor.hpp"
#include "cuda/cuda_library.hpp"
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <mutex>
#include <thread>
// clang-format on

namespace autogen {
template <typename BaseScalar>
using ADCG = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
template <typename BaseScalar>
using ADFunctor = typename std::function<void(
    const std::vector<ADCG<BaseScalar>> &, std::vector<ADCG<BaseScalar>> &)>;

template <typename BaseScalar>
struct FunctionTrace {
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename CppAD::ADFun<CGScalar>;
  using CGAtomicFunBridge = typename CppAD::cg::CGAtomicFunBridge<BaseScalar>;

  std::string name;

  std::shared_ptr<ADFun> tape{nullptr};
  // TODO fix memory leak here (bridge cannot be destructed without triggering
  // atomic_index exception in debug mode)
  CGAtomicFunBridge *bridge{nullptr};

  std::vector<BaseScalar> trace_input;

  ADFunctor<BaseScalar> functor;
  size_t input_dim;
  size_t output_dim;
  std::vector<ADCGScalar> ax;
  std::vector<ADCGScalar> ay;

  virtual ~FunctionTrace() {
    //  delete tape;
    //  // TODO atomic_base destructor in CppAD gets called twice and throws an
    //  // exception
#ifdef NDEBUG
    // we don't trigger the CPPAD_ASSERT here
    // delete bridge;
#endif
  }
};

template <typename BaseScalar = double>
struct CodeGenData {
  /**
   * Maps each function that is executed to its trace.
   */
  static inline std::map<std::string, FunctionTrace<BaseScalar>> traces;
  /**
   * Keeps track of the order of atomic function invocations, i.e. functions
   * that are called later are added later to this list.
   */
  static inline std::vector<std::string> invocation_order;

  /**
   * Defines whether the current atomic function should record the gradient tape
   * or not.
   */
  static inline bool is_dry_run{true};

  /**
   * Maps name of the caller to the names of the (atomic) functions it executes.
   */
  static inline std::map<std::string, std::vector<std::string>> call_hierarchy;

  static void clear() {
    traces.clear();
    invocation_order.clear();
    call_hierarchy.clear();
  }

  CodeGenData() = delete;
};

template <typename Scalar = double>
struct GeneratedCppAD {
  using CppADScalar = typename CppAD::AD<Scalar>;
  using Functor = std::function<void(const std::vector<CppADScalar> &,
                                     std::vector<CppADScalar> &)>;
  const std::string name;
  Functor f_cppad_;

 public:
  GeneratedCppAD(const std::string &name, const Functor &functor) : name(name) {
    f_cppad_ = functor;
  }

  void operator()(const std::vector<Scalar> &input,
                  std::vector<Scalar> &output) {
    conditionally_compile(input, output);
    output = tape_->Forward(0, input);
  }

  void jacobian(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               " has not yet been compiled. You need to call "
                               "the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }
    output = tape_->Jacobian(input);
    return;
  }

 protected:
  // used by CppAD
  std::shared_ptr<CppAD::ADFun<Scalar>> tape_{nullptr};
  std::vector<CppADScalar> ax_;
  std::vector<CppADScalar> ay_;

  bool is_compiled() const {
    return bool(tape_) && !ax_.empty() && !ay_.empty();
  }

  void conditionally_compile(const std::vector<Scalar> &input,
                             std::vector<Scalar> &output) {
    if (is_compiled()) {
      return;
    }
    tape_ = std::make_shared<CppAD::ADFun<Scalar>>();
    ax_.resize(input.size());
    ay_.resize(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      ax_[i] = CppADScalar(input[i]);
    }
    CppAD::Independent(ax_);
    f_cppad_(ax_, ay_);
    tape_->Dependent(ax_, ay_);
  }
};

}  // namespace autogen