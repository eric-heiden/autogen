#pragma once

#include <map>

#include "cppadcg_system.h"

// #define DEBUG 1

namespace autogen {
template <typename BaseScalar>
using ADCG = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
template <typename BaseScalar>
using ADFunctor = typename std::function<void(
    const std::vector<ADCG<BaseScalar>> &, std::vector<ADCG<BaseScalar>> &)>;

struct FunctionTrace {
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename CppAD::ADFun<CGScalar>;
  using CGAtomicFunBridge = typename CppAD::cg::CGAtomicFunBridge<BaseScalar>;

  std::string name;

  ADFun *tape{nullptr};
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
    // delete bridge;
#ifdef NDEBUG
    // we don't trigger the CPPAD_ASSERT here
    // delete bridge;
#endif
  }

  bool has_tape() const { return tape != nullptr; }
};

class CodeGenData {
 private:
  std::map<std::string, FunctionTrace> traces_;
  std::vector<std::string> invocation_order_;
  std::vector<std::string> invocation_stack_;
  bool is_dry_run_{true};
  std::map<std::string, std::vector<std::string>> call_hierarchy_;

  CodeGenData() = default;

 public:
  ~CodeGenData() = default;

  static inline CodeGenData *instance = nullptr;

  static void create() {
    if (instance == nullptr) {
      instance = new CodeGenData();
    }
  }

  /**
   * Maps each function that is executed to its trace.
   */
  static std::map<std::string, FunctionTrace> &traces() {
    if (!instance) {
      instance = new CodeGenData();
    }
    return instance->traces_;
  }

  static bool has_trace(const std::string &name) {
    return traces().find(name) != traces().end();
  }

  /**
   * Keeps track of the order of atomic function invocations, i.e. functions
   * that are called later are added later to this list.
   */
  static std::vector<std::string> &invocation_order() {
    if (!instance) {
      instance = new CodeGenData();
    }
    return instance->invocation_order_;
  }

  /**
   * Keeps track of the order of the currently executed function.
   */
  static std::vector<std::string> &invocation_stack() {
    if (!instance) {
      instance = new CodeGenData();
    }
    return instance->invocation_stack_;
  }

  /**
   * Defines whether the current atomic function should record the gradient tape
   * or not.
   */
  static bool &is_dry_run() {
    if (!instance) {
      instance = new CodeGenData();
    }
    return instance->is_dry_run_;
  }

  /**
   * Maps name of the caller to the names of the (atomic) functions it executes.
   */
  static std::map<std::string, std::vector<std::string>> &call_hierarchy() {
    if (!instance) {
      instance = new CodeGenData();
    }
    return instance->call_hierarchy_;
  }

  static void clear() {
    delete instance;
    instance = nullptr;
  }
};

template <typename BaseScalar = double>
inline void call_atomic(const std::string &name, ADFunctor<BaseScalar> functor,
                        const std::vector<ADCG<BaseScalar>> &input,
                        std::vector<ADCG<BaseScalar>> &output) {
  using ADFun = typename FunctionTrace::ADFun;
  using CGAtomicFunBridge = typename FunctionTrace::CGAtomicFunBridge;

  auto &traces = CodeGenData::traces();

#if DEBUG
  std::cout << "Calling atomic function \"" << name << "\". is_dry_run? "
            << std::boolalpha << CodeGenData::is_dry_run() << "\n";
#endif

  if (traces.find(name) == traces.end()) {
    auto &order = CodeGenData::invocation_order();
    auto &stack = CodeGenData::invocation_stack();
    if (!stack.empty()) {
      // the current function is called by another function, hence update the
      // call hierarchy
      const std::string &parent = stack.back();
      auto &hierarchy = CodeGenData::call_hierarchy();
      if (hierarchy.find(parent) == hierarchy.end()) {
        hierarchy[parent] = std::vector<std::string>();
      }
      hierarchy[parent].push_back(name);
    }
    order.push_back(name);
    // std::cout << " order size: " << order.size() << std::endl;

    stack.push_back(name);
    FunctionTrace trace;
    trace.name = name;
    trace.functor = functor;
    trace.trace_input.resize(input.size());
    trace.input_dim = static_cast<int>(input.size());
    trace.output_dim = static_cast<int>(output.size());
    // trace.ax.resize(input.size());
    // trace.ay.resize(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      BaseScalar raw = to_double(input[i]);
      trace.trace_input[i] = raw;
      // trace.ax[i] = ADCGScalar(raw);
    }
    // CppAD::Independent(trace.ax);
    // trace.functor(trace.ax, trace.ay);
    // trace.tape = new ADFun;
    // trace.tape->Dependent(trace.ax, trace.ay);
    // trace.tape->function_name_set(name);
    // trace.bridge = create_atomic_fun_bridge(trace.name, *(trace.tape), true);

    traces[name] = trace;
    // call this function (to discover more nested atomics)
    functor(input, output);
    // std::cout << "Traced atomic function \"" << trace.name << "\".\n";
#if DEBUG
    std::cout << "\tNew function trace created.\n";
#endif
    // std::cout << "Invocation order: ";
    // for (const auto &s : *CodeGenData::invocation_order()) {
    //   std::cout << s << " ";
    // }
    // std::cout << std::endl;
    stack.pop_back();  // remove current function from the stack
    return;
  } else if (CodeGenData::is_dry_run()) {
#if DEBUG
    std::cout << "\tAlready traced during this dry run.\n";
#endif
    // std::cout << "Invocation order: ";
    // for (const auto &s : *CodeGenData::invocation_order()) {
    //   std::cout << s << " ";
    // }
    // std::cout << std::endl;
    // we already preprocessed this function during the dry run
    return;
  }

#if DEBUG
  std::cout << "\tCalling existing function trace.\n";
#endif

  FunctionTrace &trace = traces[name];
  // std::cout << "\tTrace bridge: " << trace.bridge << std::endl;
  if (!trace.bridge) {
    throw std::runtime_error(
        "CGAtomicFunBridge for atomic function \"" + name +
        "\" is missing. Make sure to call `trace_existing_atomics()`.");
  }
  // std::cout << "call_atomic_fun_bridge(trace.bridge, input, output)\n";
  call_atomic_fun_bridge(trace.bridge, input, output);
  // (*(trace.bridge))(input, output);
  // std::cout << "Done call_atomic_fun_bridge" << std::endl;

  // std::cout << "Invocation order: ";
  // for (const auto &s : *CodeGenData::invocation_order()) {
  //   std::cout << s << " ";
  // }
  // std::cout << std::endl;
}

inline void trace_existing_atomics() {
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename CppAD::ADFun<CGScalar>;

  const auto &order = CodeGenData::invocation_order();
#ifdef DEBUG
  std::cout << "invocation order: ";
  for (const auto &s : order) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
#endif
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    if (!CodeGenData::has_trace(*it)) {
      continue;
    }
    // for (auto &[name, trace] : CodeGenData::traces()) {
    FunctionTrace &trace = CodeGenData::traces()[*it];
    if (trace.bridge) {
      std::cout << "\tFunction \"" << trace.name
                << "\" already has a bridge.\n";
      continue;
    }
    std::cout << "\tTracing atomic function \"" << trace.name
              << "\" for code generation...\n";

    std::vector<ADCGScalar> ax(trace.input_dim);
    std::vector<ADCGScalar> ay(trace.output_dim);
    // trace.ax.resize(trace.input_dim);
    // trace.ay.resize(trace.output_dim);
    // std::cout << "\tADCG input:";
    for (size_t i = 0; i < trace.input_dim; ++i) {
      ax[i] = ADCGScalar(trace.trace_input[i]);
      // std::cout << "  " << ax[i];
    }
    CppAD::Independent(ax);
    trace.functor(ax, ay);
    trace.tape = new ADFun(ax, ay);
    // trace.tape->Dependent(ax, ay);
    trace.ax = ax;
    trace.ay = ay;
    trace.tape->function_name_set(trace.name);
    // std::cout << "\tTape \"" << trace.name << "\" has " <<
    // trace.tape->size_op()
    //           << " operator(s) in the operation sequence.\n";
#ifdef DEBUG
    {
      std::cout << "\t\tTesting \"" << trace.name
                << "\" tape code with
          input : "; typedef CppAD::cg::CG<BaseScalar> CGScalar;
                  std::vector<CGScalar>
                      indVars(trace.input_dim);
      // const auto& x = model_source_gen_->getTypicalIndependentValues();
      // if (x.size() > 0) {
      for (std::size_t i = 0; i < indVars.size(); i++) {
        indVars[i].setValue(trace.trace_input[i]);
        std::cout << "  " << indVars[i];
      }
      // }
      std::cout << "\n";
      std::vector<CGScalar> dep;
      dep = trace.tape->Forward(0, indVars);
      std::cout << "\t\tTesting \"" << trace.name << "\" tape code output:";
      for (const auto &d : dep) {
        std::cout << "  " << d;
      }
      std::cout << "\n";
      std::cout << "\t\tActual \"" << trace.name << "\" tape output:";
      for (const auto &d : trace.ay) {
        std::cout << "  " << d;
      }
      std::cout << "\n\n";
    }

    {
      std::cout << "\tTesting \"" << trace.name
                << "\" tape code with random input:";
      srand(123);
      typedef CppAD::cg::CG<BaseScalar> CGScalar;
      std::vector<CGScalar> indVars(trace.input_dim);
      // const auto& x = model_source_gen_->getTypicalIndependentValues();
      // if (x.size() > 0) {
      for (std::size_t i = 0; i < indVars.size(); i++) {
        indVars[i].setValue(double(rand()) / RAND_MAX);
        std::cout << "  " << indVars[i];
      }
      // }
      std::cout << "\n";
      std::vector<CGScalar> dep;
      dep = trace.tape->Forward(0, indVars);
      std::cout << "\t\tTesting \"" << trace.name << "\" tape code output:";
      for (const auto &d : dep) {
        std::cout << "  " << d;
      }
      std::cout << "\n";
    }
#endif

    trace.bridge = create_atomic_fun_bridge(trace.name, *(trace.tape), true);
  }
  // std::cout << "### Done tracing existing atomics in C++\n\n\n";
}

template <typename Functor>
inline FunctionTrace trace(Functor functor, const std::string &name,
                           const std::vector<BaseScalar> &input,
                           std::vector<BaseScalar> &output) {
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename CppAD::ADFun<CGScalar>;

  CodeGenData::clear();

  // first, a "dry run" to discover the atomic functions
  {
    CodeGenData::is_dry_run() = true;
    std::vector<ADCGScalar> ax(input.size()), ay(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      ax[i] = ADCGScalar(to_double(input[i]));
    }
    functor(ax, ay);
    CodeGenData::is_dry_run() = false;
  }

  // next, trace the inner atomic functions
  trace_existing_atomics();

  // finally, trace the top-level function
  FunctionTrace trace;
  trace.name = name;
  std::vector<ADCGScalar> ax(input.size()), ay(output.size());
  std::cout << "Tracing function \"" << name << "\" for code generation...\n";
  for (size_t i = 0; i < input.size(); ++i) {
    ax[i] = ADCGScalar(to_double(input[i]));
  }
  CppAD::Independent(ax);
  functor(ax, ay);
  trace.tape = new ADFun;
  trace.tape->Dependent(ax, ay);
  trace.tape->function_name_set(name);
  trace.bridge = create_atomic_fun_bridge(name, *(trace.tape), true);
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