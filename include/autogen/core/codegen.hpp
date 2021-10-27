#pragma once

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>
#include <map>
#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif

#include "base.hpp"
#include "types.h"

// #define DEBUG 1

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
  int input_dim;
  int output_dim;
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
  static inline std::map<std::string, FunctionTrace<BaseScalar>> *traces =
      new std::map<std::string, FunctionTrace<BaseScalar>>;
  /**
   * Keeps track of the order of atomic function invocations, i.e. functions
   * that are called later are added later to this list.
   */
  static inline std::vector<std::string> *invocation_order =
      new std::vector<std::string>;
  /**
   * Keeps track of the order of the currently executed function.
   */
  static inline std::vector<std::string> *invocation_stack =
      new std::vector<std::string>;

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
    traces->clear();
    invocation_order->clear();
    call_hierarchy.clear();
    invocation_stack->clear();
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

#if DEBUG
  std::cout << "Calling atomic function \"" << name << "\". is_dry_run? "
            << std::boolalpha << CodeGenData<BaseScalar>::is_dry_run << "\n";
#endif

  if (traces->find(name) == traces->end()) {
    auto &order = *CodeGenData<BaseScalar>::invocation_order;
    auto &stack = *CodeGenData<BaseScalar>::invocation_stack;
    if (!stack.empty()) {
      // the current function is called by another function, hence update the
      // call hierarchy
      const std::string &parent = stack.back();
      auto &hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
      if (hierarchy.find(parent) == hierarchy.end()) {
        hierarchy[parent] = std::vector<std::string>();
      }
      hierarchy[parent].push_back(name);
    }
    order.push_back(name);
    stack.push_back(name);
    FunctionTrace<BaseScalar> trace;
    trace.name = name;
    trace.functor = functor;
    trace.trace_input.resize(input.size());
    trace.input_dim = static_cast<int>(input.size());
    trace.output_dim = static_cast<int>(output.size());
    trace.ax.resize(input.size());
    trace.ay.resize(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      BaseScalar raw = to_double(input[i]);
      trace.trace_input[i] = raw;
      trace.ax[i] = ADCGScalar(raw);
    }

    CppAD::Independent(trace.ax);
    trace.functor(trace.ax, trace.ay);
    trace.tape = std::make_shared<ADFun>();
    trace.tape->Dependent(trace.ax, trace.ay);
    trace.tape->function_name_set(name);
    trace.bridge = new CGAtomicFunBridge(trace.name, *(trace.tape), true);

    (*traces)[name] = trace;
    functor(input, output);
    std::cout << "Traced atomic function \"" << trace.name << "\".\n";
#if DEBUG
    std::cout << "\tNew function trace created.\n";
#endif
    // std::cout << "Invocation order: ";
    // for (const auto &s : *CodeGenData<BaseScalar>::invocation_order) {
    //   std::cout << s << " ";
    // }
    // std::cout << std::endl;    
    stack.pop_back();  // remove current function from the stack
    return;
  } else if (CodeGenData<BaseScalar>::is_dry_run) {
#if DEBUG
    std::cout << "\tAlready traced during this dry run.\n";
#endif
    // std::cout << "Invocation order: ";
    // for (const auto &s : *CodeGenData<BaseScalar>::invocation_order) {
    //   std::cout << s << " ";
    // }
    // std::cout << std::endl;
    // we already preprocessed this function during the dry run
    return;
  }

#if DEBUG
  std::cout << "\tCalling existing function trace.\n";
#endif

  FunctionTrace<BaseScalar> &trace = (*traces)[name];
  if (!trace.bridge) {
    throw std::runtime_error("CGAtomicFunBridge for atomic function \"" + name +
                             "\" is missing");
  }
  (*(trace.bridge))(input, output);

  // std::cout << "Invocation order: ";
  // for (const auto &s : *CodeGenData<BaseScalar>::invocation_order) {
  //   std::cout << s << " ";
  // }
  // std::cout << std::endl;
}

template <typename Functor>
static FunctionTrace<BaseScalar> trace(Functor functor, const std::string &name,
                                       const std::vector<BaseScalar> &input,
                                       std::vector<BaseScalar> &output) {
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename CppAD::ADFun<CGScalar>;
  using CGAtomicFunBridge = typename CppAD::cg::CGAtomicFunBridge<BaseScalar>;

  CodeGenData<BaseScalar>::clear();

  // first, a "dry run" to discover the atomic functions
  {
    CodeGenData<BaseScalar>::is_dry_run = true;
    std::vector<ADCGScalar> ax(input.size()), ay(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      ax[i] = ADCGScalar(to_double(input[i]));
    }
    functor(ax, ay);
    CodeGenData<BaseScalar>::is_dry_run = false;
  }

  // next, trace the inner atomic functions
  const auto &order = *CodeGenData<BaseScalar>::invocation_order;
  std::cout << "Function \"" << name << "\" has " << order.size()
            << " atomic function(s).\n";
  // TODO can this be removed?
  // for (auto it = order.rbegin(); it != order.rend(); ++it) {
  //   FunctionTrace<BaseScalar> &trace = (*CodeGenData<BaseScalar>::traces)[*it];
  //   std::cout << "Tracing function \"" << trace.name
  //             << "\" for code generation...\n";
  //   trace.ax.resize(trace.input_dim);
  //   trace.ay.resize(trace.output_dim);
  //   for (size_t i = 0; i < trace.input_dim; ++i) {
  //     trace.ax[i] = ADCGScalar(to_double(trace.trace_input[i]));
  //   }
  //   CppAD::Independent(trace.ax);
  //   trace.functor(trace.ax, trace.ay);
  //   trace.tape = std::make_shared<ADFun>();
  //   trace.tape->Dependent(trace.ax, trace.ay);
  //   trace.tape->function_name_set(trace.name);
  //   trace.bridge = new CGAtomicFunBridge(trace.name, *(trace.tape), true);
  // }

  // finally, trace the top-level function
  FunctionTrace<BaseScalar> trace;
  trace.name = name;
  std::vector<ADCGScalar> ax(input.size()), ay(output.size());
  std::cout << "Tracing function \"" << name << "\" for code generation...\n";
  for (size_t i = 0; i < input.size(); ++i) {
    ax[i] = ADCGScalar(to_double(input[i]));
  }
  CppAD::Independent(ax);
  functor(ax, ay);
  trace.tape = std::make_shared<ADFun>();
  trace.tape->Dependent(ax, ay);
  trace.tape->function_name_set(name);
  trace.bridge = new CGAtomicFunBridge(name, *(trace.tape), true);
  trace.input_dim = static_cast<int>(input.size());
  trace.output_dim = static_cast<int>(output.size());
  return trace;
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