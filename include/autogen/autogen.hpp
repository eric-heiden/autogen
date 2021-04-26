#pragma once

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>
#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif

// clang-format off
#include "utils/conditionals.hpp"
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

template <template <typename> typename Functor, typename BaseScalar = double>
struct Generated {
  static inline std::map<std::string, FunctionTrace<BaseScalar>> traces;

  const std::string name;

  using CppADScalar = typename CppAD::AD<BaseScalar>;
  using CGScalar = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;

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

 protected:
  std::unique_ptr<Functor<BaseScalar>> f_double_;
  std::unique_ptr<Functor<CppADScalar>> f_cppad_;
  std::unique_ptr<Functor<CGScalar>> f_cg_;

  GenerationMode mode_{GENERATE_CPU};
  AccumulationMethod jac_acc_method_{ACCUMULATE_NONE};

  mutable std::mutex compilation_mutex_;
  bool is_compiling_{false};

  // name of the compiled library
  std::string library_name_;

  size_t local_input_dim_{0};
  size_t global_input_dim_{0};
  size_t output_dim_{0};

  // used by CppAD (only in GENERATE_CPPAD mode)
  std::shared_ptr<CppAD::ADFun<BaseScalar>> tape_{nullptr};
  std::vector<CppADScalar> ax_;
  std::vector<CppADScalar> ay_;

 public:
  template <typename... Args>
  Generated(const std::string &name, Args &&... args) : name(name) {
    f_double_ =
        std::make_unique<Functor<BaseScalar>>(std::forward<Args>(args)...);
    f_cppad_ =
        std::make_unique<Functor<CppADScalar>>(std::forward<Args>(args)...);
    f_cg_ = std::make_unique<Functor<CGScalar>>(std::forward<Args>(args)...);
  }

  // discards the compiled library (so that it gets recompiled at the next
  // evaluation)
  void discard_library() {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    library_name_ = "";
  }

  GenerationMode mode() const { return mode_; }
  void set_mode(GenerationMode mode) {
    if (mode != this->mode_) {
      // changing the mode discards the previously compiled library
      discard_library();
    }
    this->mode_ = mode;
  }

  AccumulationMethod jacobian_acc_method() const { return jac_acc_method_; }
  void set_jacobian_acc_method(AccumulationMethod jac_acc_method) {
    if (jac_acc_method != this->jac_acc_method_) {
      // changing the jac_acc_method discards the previously compiled library
      discard_library();
    }
    this->jac_acc_method_ = jac_acc_method;
  }

  size_t input_dim() const { return local_input_dim_ + global_input_dim_; }
  size_t local_input_dim() const { return local_input_dim_; }
  size_t output_dim() const { return output_dim_; }

  size_t global_input_dim() const { return global_input_dim_; }
  void set_global_input_dim(size_t global_input_dim) {
    if (global_input_dim != global_input_dim_) {
      discard_library();
    }
    global_input_dim_ = global_input_dim;
  }

  bool is_compiled() const {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    switch (mode_) {
      case GENERATE_NONE:
        return true;
      case GENERATE_CPPAD:
        return bool(tape_) && !ax_.empty() && !ay_.empty();
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

  void compile(const FunctionTrace<BaseScalar> &main_trace) {
    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = true;
    }

    if (mode_ == GENERATE_CPU) {
      compile_cpu(main_trace);
    } else if (mode_ == GENERATE_CUDA) {
      compile_cuda(main_trace);
    }

    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = false;
    }
  }

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) {
    conditionally_compile(input, output);

    if (mode_ == GENERATE_NONE) {
      (*f_double_)(input, output);
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
      const auto &model = get_cuda_model();
      model.forward_zero(input, output);
    } else if (mode_ == GENERATE_CPPAD) {
      output = tape_->Forward(0, input);
    }
  }

  /**
   * Vectorized execution of the forward pass of this function, which will
   * compute the outputs for each of the inputs in parallel.
   */
  void operator()(const std::vector<std::vector<BaseScalar>> &local_inputs,
                  std::vector<std::vector<BaseScalar>> &outputs,
                  const std::vector<BaseScalar> &global_input = {}) {
    if (local_inputs.empty()) {
      return;
    }
    outputs.resize(local_inputs.size());

    conditionally_compile(local_inputs, outputs, global_input);

    if (mode_ == GENERATE_NONE || mode_ == GENERATE_CPPAD) {
      if (global_input.empty()) {
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          if (mode_ == GENERATE_NONE) {
            (*f_double_)(local_inputs[i], outputs[i]);
          } else {
            outputs[i] = tape_->Forward(0, local_inputs[i]);
          }
        }
      } else {
        std::vector<BaseScalar> input(global_input);
        input.resize(global_input.size() + local_inputs[0].size());
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          for (size_t j = 0; j < local_inputs[i].size(); ++j) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          if (mode_ == GENERATE_NONE) {
            (*f_double_)(input, outputs[i]);
          } else {
            outputs[i] = tape_->Forward(0, input);
          }
        }
      }
    } else if (mode_ == GENERATE_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      for (auto &o : outputs) {
        o.resize(output_dim_);
      }
      int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
      for (int i = 0; i < num_tasks; ++i) {
        if (global_input.empty()) {
          GenericModelPtr &model = get_cpu_model();
          model->ForwardZero(local_inputs[i], outputs[i]);
        } else {
          static thread_local std::vector<BaseScalar> input;
          input = global_input;
          input.resize(global_input.size() + local_inputs[0].size());
          for (size_t j = 0; j < local_inputs[i].size(); ++i) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          GenericModelPtr &model = get_cpu_model();
          model->ForwardZero(input, outputs[i]);
        }
      }
#endif
    } else if (mode_ == GENERATE_CUDA) {
      const auto &model = get_cuda_model();
      model.forward_zero(&outputs, local_inputs, num_gpu_threads_per_block,
                         global_input);
    }
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) {
    conditionally_compile(input, output);
    if (mode_ == GENERATE_NONE) {
      // central difference
      assert(output_dim() > 0);
      output.resize(input_dim() * output_dim());
      std::vector<BaseScalar> left_x = input, right_x = input;
      std::vector<BaseScalar> left_y(output_dim()), right_y(output_dim());
      for (size_t i = 0; i < input.size(); ++i) {
        left_x[i] -= finite_diff_eps;
        right_x[i] += finite_diff_eps;
        BaseScalar dx = right_x[i] - left_x[i];
        (*f_double_)(left_x, left_y);
        (*f_double_)(right_x, right_y);
        for (size_t j = 0; j < output_dim(); ++j) {
          output[j * input_dim() + i] = (right_y[j] - left_y[j]) / dx;
        }
        left_x[i] = right_x[i] = input[i];
      }
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      output = tape_->Jacobian(input);
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
      const auto &model = get_cuda_model();
      model.jacobian(input, output);
    }
  }

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input = {}) {
    outputs.resize(local_inputs.size());
    conditionally_compile(local_inputs, outputs, global_input);
    if (mode_ == GENERATE_NONE || mode_ == GENERATE_CPPAD) {
      if (global_input.empty()) {
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          jacobian(local_inputs[i], outputs[i]);
        }
      } else {
        std::vector<BaseScalar> input(global_input.size());
        input.insert(input.begin(), global_input.begin(), global_input.end());
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          for (size_t j = 0; j < local_inputs[i].size(); ++i) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          jacobian(input, outputs[i]);
        }
      }
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
      for (auto &o : outputs) {
        o.resize(output_dim_);
      }
      int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
      for (int i = 0; i < num_tasks; ++i) {
        if (global_input.empty()) {
          GenericModelPtr &model = get_cpu_model();
          model->ForwardZero(local_inputs[i], outputs[i]);
          model->Jacobian(local_inputs[i], outputs[i]);
        } else {
          static thread_local std::vector<BaseScalar> input;
          if (input.empty()) {
            input.resize(global_input.size());
            input.insert(input.begin(), global_input.begin(),
                         global_input.end());
          }
          for (size_t j = 0; j < local_inputs[i].size(); ++i) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          GenericModelPtr &model = get_cpu_model();
          model->Jacobian(input, outputs[i]);
        }
      }
#endif
    } else if (mode_ == GENERATE_CUDA) {
      const auto &model = get_cuda_model();
      model.jacobian(&outputs, local_inputs, num_gpu_threads_per_block,
                     global_input);
    }
  }

 protected:
  void conditionally_compile(const std::vector<BaseScalar> &input,
                             std::vector<BaseScalar> &output) {
    if (input_dim() == 0 || output_dim() == 0) {
      // retrieve dimensions by evaluating double-instantiated functor on
      // provided input
      (*f_double_)(input, output);
      local_input_dim_ = input.size();
      output_dim_ = output.size();
    }
    if (is_compiled()) {
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
      ax_.resize(input.size());
      ay_.resize(output.size());
      for (size_t i = 0; i < input.size(); ++i) {
        ax_[i] = CppADScalar(input[i]);
      }
      CppAD::Independent(ax_);
      (*f_cppad_)(ax_, ay_);
      tape_->Dependent(ax_, ay_);
      return;
    }
    if (mode_ == GENERATE_CPU || mode_ == GENERATE_CUDA) {
      assert(!input.empty());
      assert(!output.empty());
      FunctionTrace<BaseScalar> t = trace(input, output);
      if (compile_in_background) {
        std::thread worker([this, &t]() { compile(t); });
        (*f_double_)(input, output);
        return;
      } else {
        compile(t);
        std::cout << "Finished compilation.\n";
      }
    }
  }

  void conditionally_compile(
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      std::vector<std::vector<BaseScalar>> &outputs,
      const std::vector<BaseScalar> &global_input) {
    global_input_dim_ = global_input.size();
    std::vector<BaseScalar> compilation_input;
    compilation_input.insert(compilation_input.end(), global_input.begin(),
                             global_input.end());
    compilation_input.insert(compilation_input.end(), local_inputs[0].begin(),
                             local_inputs[0].end());
    conditionally_compile(compilation_input, outputs[0]);
  }

  FunctionTrace<BaseScalar> trace(const std::vector<BaseScalar> &input,
                                  std::vector<BaseScalar> &output) const {
    CodeGenData<BaseScalar>::clear();

    // first, a "dry run" to discover the atomic functions
    {
      CodeGenData<BaseScalar>::is_dry_run = true;
      std::vector<CGScalar> ax(input.size()), ay(output.size());
      for (size_t i = 0; i < input.size(); ++i) {
        ax[i] = CGScalar(input[i]);
      }
      (*f_cg_)(ax, ay);
      CodeGenData<BaseScalar>::is_dry_run = false;
    }

    // next, trace the inner atomic functions
    const auto &order = CodeGenData<BaseScalar>::invocation_order;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      FunctionTrace<BaseScalar> &trace = CodeGenData<BaseScalar>::traces[*it];
      std::cout << "Tracing function \"" << trace.name
                << "\" for code generation...\n";
      trace.ax.resize(trace.input_dim);
      trace.ay.resize(trace.output_dim);
      for (size_t i = 0; i < trace.input_dim; ++i) {
        trace.ax[i] = CGScalar(trace.trace_input[i]);
      }
      CppAD::Independent(trace.ax);
      trace.functor(trace.ax, trace.ay);
      trace.tape = std::make_shared<ADFun>();
      trace.tape->Dependent(trace.ax, trace.ay);
      trace.bridge = new CGAtomicFunBridge(trace.name, *(trace.tape), true);
    }

    // finally, trace the top-level function
    FunctionTrace<BaseScalar> trace;
    trace.name = name;
    std::vector<CGScalar> ax(input.size()), ay(output.size());
    std::cout << "Tracing function \"" << name << "\" for code generation...\n";
    for (size_t i = 0; i < input.size(); ++i) {
      ax[i] = CGScalar(input[i]);
    }
    CppAD::Independent(ax);
    (*f_cg_)(ax, ay);
    trace.tape = std::make_shared<ADFun>();
    trace.tape->Dependent(ax, ay);
    trace.bridge = new CGAtomicFunBridge(name, *(trace.tape), true);
    trace.input_dim = input.size();
    trace.output_dim = output.size();
    return trace;
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

  void compile_cuda(const FunctionTrace<BaseScalar> &main_trace) {
    using namespace CppAD;
    using namespace CppAD::cg;

    CudaModelSourceGen<BaseScalar> main_source_gen(*(main_trace.tape), name);
    main_source_gen.setCreateJacobian(true);
    main_source_gen.global_input_dim() = global_input_dim_;
    main_source_gen.jacobian_acc_method() = jac_acc_method_;
    CudaLibraryProcessor<BaseScalar> cuda_proc(&main_source_gen,
                                               name + "_cuda");
    // reverse order of invocation to first generate code for innermost
    // functions
    const auto &order = CodeGenData<BaseScalar>::invocation_order;
    std::list<CudaModelSourceGen<BaseScalar> *> models;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      std::cout << "Adding cuda model " << *it << "\n";
      FunctionTrace<BaseScalar> &trace = CodeGenData<BaseScalar>::traces[*it];
      auto *source_gen = new CudaModelSourceGen<BaseScalar>(*(trace.tape), *it);
      source_gen->setCreateForwardOne(true);
      source_gen->setCreateReverseOne(true);
      source_gen->set_kernel_only(true);
      models.push_back(source_gen);
      cuda_proc.add_model(models.back(), false);
    }
    cuda_proc.debug_mode() = debug_mode;
    cuda_proc.generate_code();
    cuda_proc.save_sources();
    cuda_proc.create_library();

    std::lock_guard<std::mutex> guard(compilation_mutex_);
    library_name_ = name + "_cuda";

    for (auto *model : models) {
      delete model;
    }
  }

  const CudaModel<BaseScalar> &get_cuda_model() const {
    static thread_local const CudaLibrary<BaseScalar> library(library_name_);
    static thread_local const CudaModel<BaseScalar> &model =
        library.get_model(name);
    return model;
  }
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