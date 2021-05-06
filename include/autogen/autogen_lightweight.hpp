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

enum GenerationMode {
  GENERATE_NONE,
  GENERATE_CPPAD,
  GENERATE_CPU,
  GENERATE_CUDA
};

static inline std::string str(const GenerationMode &mode) {
  switch (mode) {
    case GENERATE_NONE:
      return "None";
    case GENERATE_CPPAD:
      return "CppAD";
    case GENERATE_CPU:
      return "CPU";
    case GENERATE_CUDA:
      return "CUDA";
  }
  return "Unknown";
}

static inline std::ostream &operator<<(std::ostream &os,
                                       const GenerationMode &mode) {
  os << str(mode);
  return os;
}

template <typename BaseScalar = double>
struct GeneratedLightWeight {
  using ADCG = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;

 public:
  GeneratedLightWeight(const std::string &name, std::shared_ptr<ADFun> ad_fun)
      : name(name) {
    ad_fun_ = ad_fun;
  }

  GenerationMode mode() const { return mode_; }
  void set_mode(GenerationMode mode) {
    if (mode != this->mode_) {
      // changing the mode discards the previously compiled library
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      library_name_ = "";
    }
    this->mode_ = mode;
  }

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) {
    conditionally_compile(input, output);

    if (mode_ == GENERATE_NONE) {
      //      (*f_double_)(input, output);
    } else if (mode_ == GENERATE_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      GenericModelPtr &model = get_cpu_model();
      model->ForwardZero(input, output);
#endif
    } else if (mode_ == GENERATE_CUDA) {
      //      const auto &model = get_cuda_model();
      //      model.forward_zero(input, output);
    } else if (mode_ == GENERATE_CPPAD) {
      //      output = tape_->Forward(0, input);
    }
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) {
    // TODO conditionally compile? (need to know output dim)
    if (mode_ == GENERATE_NONE) {
      //      // central difference
      //      assert(input.size() == input_dim());
      //      assert(output_dim() > 0);
      //      output.resize(input_dim() * output_dim());
      //      std::vector<BaseScalar> left_x = input, right_x = input;
      //      std::vector<BaseScalar> left_y(output_dim()),
      //      right_y(output_dim()); for (size_t i = 0; i < input.size(); ++i) {
      //        left_x[i] -= finite_diff_eps;
      //        right_x[i] += finite_diff_eps;
      //        BaseScalar dx = right_x[i] - left_x[i];
      //        (*f_double_)(left_x, left_y);
      //        (*f_double_)(right_x, right_y);
      //        for (size_t j = 0; j < output_dim(); ++j) {
      //          output[j * input_dim() + i] = (right_y[j] - left_y[j]) / dx;
      //        }
      //        left_x[i] = right_x[i] = input[i];
      //      }
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      //      output = tape_->Jacobian(input);
      return;
    }

    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               "\" has not yet been compiled in " + str(mode_) +
                               " mode. You need to call the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }

    if (mode_ == GENERATE_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      GenericModelPtr &model = get_cpu_model();
      model->Jacobian(input, output);
#endif
    } else if (mode_ == GENERATE_CUDA) {
      //      const auto &model = get_cuda_model();
      //      model.jacobian(input, output);
    }
  }

 protected:
  const std::string name;
  std::shared_ptr<ADFun> ad_fun_;

  GenerationMode mode_{GENERATE_CPU};
  mutable std::mutex compilation_mutex_;
  bool is_compiling_{false};
  typedef std::unique_ptr<CppAD::cg::GenericModel<BaseScalar>> GenericModelPtr;
  bool compile_in_background{false};
  int num_gpu_threads_per_block{32};
  /**
   * Step size to use for finite differencing.
   */
  double finite_diff_eps{1e-6};
  /**
   * Whether the generated code is compiled in debug mode (only applies to CPU
   * and CUDA).
   */
  bool debug_mode{false};
  // name of the compiled library
  std::string library_name_;

  size_t local_input_dim_{0};
  size_t global_input_dim_{0};
  size_t output_dim_{0};

  // used by CppAD (only in GENERATE_CPPAD mode)
  //  std::shared_ptr<CppAD::ADFun<BaseScalar>> tape_{nullptr};
  //  std::vector<CppADScalar> ax_;
  //  std::vector<CppADScalar> ay_;

  bool is_compiled() const {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    switch (mode_) {
      case GENERATE_NONE:
        return true;
      case GENERATE_CPPAD:
        //        return bool(tape_) && !ax_.empty() && !ay_.empty();
      case GENERATE_CPU:
      case GENERATE_CUDA:
        return !library_name_.empty();
    }
    return false;
  }

  bool is_compiling() const {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    return is_compiling_;
  }

  void conditionally_compile(const std::vector<BaseScalar> &input,
                             std::vector<BaseScalar> &output) {
    if (is_compiled()) {
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      //      tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
      //      ax_.resize(input.size());
      //      ay_.resize(output.size());
      //      for (size_t i = 0; i < input.size(); ++i) {
      //        ax_[i] = CppADScalar(input[i]);
      //      }
      //      CppAD::Independent(ax_);
      //      (*f_cppad_)(ax_, ay_);
      //      tape_->Dependent(ax_, ay_);
      return;
    }
    if (mode_ == GENERATE_CPU || mode_ == GENERATE_CUDA) {
      assert(!input.empty());
      assert(!output.empty());
      local_input_dim_ = input.size();
      output_dim_ = output.size();
      //      FunctionTrace<BaseScalar> t = trace(input, output);
      FunctionTrace<BaseScalar> t = build_trace(ad_fun_);
      if (compile_in_background) {
        std::thread worker([this, &t]() { compile(t); });
        //        (*f_double_)(input, output);
        return;
      } else {
        compile(t);
        std::cout << "Finished compilation.\n";
      }
    }
  }

  FunctionTrace<BaseScalar> build_trace(std::shared_ptr<ADFun> ad_fun) {
    FunctionTrace<BaseScalar> trace;
    trace.name = name;
    std::cout << "Tracing function \"" << name << "\" for code generation...\n";
    trace.tape = ad_fun;
    trace.bridge = new CGAtomicFunBridge(name, *(trace.tape), true);
    // Don't know these dimensions
    //    trace.input_dim = input.size();
    //    trace.output_dim = output.size();
    //    trace.input_dim = 2;
    //    trace.output_dim = 2;
    return trace;
  }

  void compile(const FunctionTrace<BaseScalar> &main_trace) {
    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = true;
    }

    if (mode_ == GENERATE_CPU) {
      compile_cpu(main_trace);
    } else if (mode_ == GENERATE_CUDA) {
      //      compile_cuda(main_trace);
    }

    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = false;
    }
  }

  void compile_cpu(const FunctionTrace<BaseScalar> &main_trace) {
    using namespace CppAD;
    using namespace CppAD::cg;

    ModelCSourceGen<BaseScalar> main_source_gen(*(main_trace.tape), name);
    main_source_gen.setCreateJacobian(true);
    ModelLibraryCSourceGen<BaseScalar> libcgen(main_source_gen);
    // reverse order of invocation to first generate code for innermost
    // functions
    const auto &order = CodeGenData<BaseScalar>::invocation_order;
    std::list<ModelCSourceGen<BaseScalar> *> models;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      FunctionTrace<BaseScalar> &trace = CodeGenData<BaseScalar>::traces[*it];
      auto *source_gen = new ModelCSourceGen<BaseScalar>(*(trace.tape), *it);
      // source_gen->setCreateSparseJacobian(true);
      // source_gen->setCreateJacobian(true);
      source_gen->setCreateForwardOne(true);
      source_gen->setCreateReverseOne(true);
      models.push_back(source_gen);
      // we need a stable reference
      libcgen.addModel(*(models.back()));
    }
    libcgen.setVerbose(true);
#if CPPAD_CG_SYSTEM_WIN
    SaveFilesModelLibraryProcessor<double> psave(libcgen);
    psave.saveSourcesTo(name + "_cpu_srcs");
    std::cerr << "CPU code compilation is not yet available on Windows. Saved "
                 "source files to \""
              << name << "_cpu_srcs"
              << "\".\n ";
    std::exit(1);
#else
    DynamicModelLibraryProcessor<BaseScalar> p(libcgen);
    auto compiler = std::make_unique<ClangCompiler<BaseScalar>>();
    compiler->setSourcesFolder(name + "_cpu_srcs");
    compiler->setSaveToDiskFirst(true);
    compiler->addCompileFlag("-O" + std::to_string(1));
    if (debug_mode) {
      compiler->addCompileFlag("-g");
      compiler->addCompileFlag("-O0");
    }
    p.setLibraryName(name + "_cpu");
    p.createDynamicLibrary(*compiler, false);

    std::lock_guard<std::mutex> guard(compilation_mutex_);
    library_name_ = "./" + name + "_cpu.so";
#endif
  }

#if !CPPAD_CG_SYSTEM_WIN
  GenericModelPtr &get_cpu_model() const {
    static thread_local bool initialized = false;
    static thread_local auto lib =
        std::make_unique<CppAD::cg::LinuxDynamicLib<BaseScalar>>(library_name_);
    static thread_local std::map<std::string, GenericModelPtr> models;
    if (!initialized) {
      // load and wire up atomic functions in this library
      const auto &order = CodeGenData<BaseScalar>::invocation_order;
      const auto &hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
      models[name] = GenericModelPtr(lib->model(name).release());
      for (const std::string &model_name : order) {
        // we have to keep the atomic function pointers alive
        models[model_name] = GenericModelPtr(lib->model(model_name).release());
        // simply add every atomic to the top-level function
        // (because we don't have hierarchy information for the top-level
        // function)
        models[name]->addAtomicFunction(models[model_name]->asAtomic());
      }
      for (const auto &[outer, inner_models] : hierarchy) {
        auto &outer_model = models[outer];
        for (const std::string &inner : inner_models) {
          auto &inner_model = models[inner];
          outer_model->addAtomicFunction(inner_model->asAtomic());
          std::cout << "Connected atomic function \"" + inner +
                           "\" to its parent function \""
                    << outer << "\".\n";
        }
      }

      std::cout << "Loaded compiled model \"" << name << "\" from \""
                << library_name_ << "\".\n";
      initialized = true;
    }
    static thread_local GenericModelPtr &model = models[name];
    return model;
  }
#endif
};

}  // namespace autogen