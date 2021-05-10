#pragma once


// clang-format off
#include "codegen.hpp"
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
class GeneratedCodeGen : public GeneratedBase {
 private:
  using CppADScalar = typename CppAD::AD<BaseScalar>;
  using CGScalar = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;

  typedef std::unique_ptr<CppAD::cg::GenericModel<BaseScalar>> GenericModelPtr;

  bool compile_in_background{false};

  int num_gpu_threads_per_block{32};

  /**
   * Whether the generated code is compiled in debug mode (only applies to CPU
   * and CUDA).
   */
  bool debug_mode{false};

 protected:
  using GeneratedBase::global_input_dim_;
  using GeneratedBase::local_input_dim_;
  using GeneratedBase::output_dim_;

  AccumulationMethod jac_acc_method_{ACCUMULATE_NONE};

  mutable std::mutex compilation_mutex_;
  bool is_compiling_{false};

  // name of the compiled library
  std::string library_name_;

  size_t local_input_dim_{0};
  size_t global_input_dim_{0};
  size_t output_dim_{0};

 public:
  // discards the compiled library (so that it gets recompiled at the next
  // evaluation)
  void discard_library() {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    library_name_ = "";
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
}  // namespace autogen