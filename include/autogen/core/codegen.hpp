#pragma once

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>
#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif

#include "base.hpp"

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

template <typename BaseScalar = double>
inline void call_atomic(const std::string &name, ADFunctor<BaseScalar> functor,
                        const std::vector<ADCG<BaseScalar>> &input,
                        std::vector<ADCG<BaseScalar>> &output) {
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;

  auto &traces = CodeGenData<BaseScalar>::traces;

  if (traces.find(name) == traces.end()) {
    auto &order = CodeGenData<BaseScalar>::invocation_order;
    if (!order.empty()) {
      // the current function is called by another function, hence update the
      // call hierarchy
      const std::string &parent = order.back();
      auto &hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
      if (hierarchy.find(parent) == hierarchy.end()) {
        hierarchy[parent] = std::vector<std::string>();
      }
      hierarchy[parent].push_back(name);
    }
    order.push_back(name);
    FunctionTrace<BaseScalar> trace;
    trace.name = name;
    trace.functor = functor;
    trace.trace_input.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      BaseScalar raw = CppAD::Value(CppAD::Var2Par(input[i])).getValue();
      trace.trace_input[i] = raw;
    }
    trace.input_dim = input.size();
    trace.output_dim = output.size();
    traces[name] = trace;
    functor(input, output);
    return;
  } else if (CodeGenData<BaseScalar>::is_dry_run) {
    // we already preprocessed this function during the dry run
    return;
  }

  FunctionTrace<BaseScalar> &trace = traces[name];
  assert(trace.bridge);
  (*(trace.bridge))(input, output);
}

template <typename Scalar>
inline void call_atomic(
    const std::string &name,
    const std::function<void(const std::vector<Scalar> &,
                             std::vector<Scalar> &)> &functor,
    const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  // no tracing occurs since the arguments are of type double
  functor(input, output);
}

/**
 * More overloads for the atomic function to be traced:
 */

template <typename BaseScalar>
inline ADCG<BaseScalar> call_atomic(
    const std::string &name,
    const std::function<ADCG<BaseScalar>(const std::vector<ADCG<BaseScalar>> &)>
        &functor,
    const std::vector<ADCG<BaseScalar>> &input) {
  ADFunctor<BaseScalar> vec_fun =
      [functor](const std::vector<ADCG<BaseScalar>> &in,
                std::vector<ADCG<BaseScalar>> &out) { out[0] = functor(in); };
  std::vector<ADCG<BaseScalar>> output(1);
  call_atomic<BaseScalar>(name, vec_fun, input, output);
  return output[0];
}

template <typename Scalar>
inline Scalar call_atomic(
    const std::string &name,
    const std::function<Scalar(const std::vector<Scalar> &)> &functor,
    const std::vector<Scalar> &input) {
  return functor(input);
}
}  // namespace autogen