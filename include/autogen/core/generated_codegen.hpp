#pragma once

// clang-format off
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <thread>

#include "../utils/conditionals.hpp"

#include "../cuda/cuda_codegen.hpp"
#include "../cuda/cuda_library_processor.hpp"
#include "../cuda/cuda_library.hpp"

#include "codegen.hpp"
// clang-format on

namespace autogen {

enum CodeGenTarget { TARGET_CPU, TARGET_CUDA };

class GeneratedCodeGen : public GeneratedBase {
  template <template <typename> typename Functor>
  friend struct Generated;

 public:
  using CppADScalar = typename CppAD::AD<BaseScalar>;
  using CGScalar = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;

 private:
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;

  typedef std::unique_ptr<CppAD::cg::GenericModel<BaseScalar>> GenericModelPtr;

  CodeGenTarget target_{TARGET_CUDA};

 protected:
  using GeneratedBase::global_input_dim_;
  using GeneratedBase::local_input_dim_;
  using GeneratedBase::output_dim_;

  AccumulationMethod jac_acc_method_{ACCUMULATE_NONE};

  // name of the compiled library
  std::string library_name_;

  std::string name_;
  FunctionTrace<BaseScalar> main_trace_;

  mutable std::shared_ptr<CudaLibrary<BaseScalar>> cuda_library_{nullptr};

#if !CPPAD_CG_SYSTEM_WIN
  mutable std::shared_ptr<CppAD::cg::LinuxDynamicLib<BaseScalar>> cpu_library_{
      nullptr};
  mutable std::map<std::string, GenericModelPtr> cpu_models_;
#endif

 public:
  int num_gpu_threads_per_block{32};

  /**
   * Whether the generated code is compiled in debug mode (only applies to CPU
   * and CUDA).
   */
  bool debug_mode{false};

  /**
   * Optimization level to use when compiling CPU and CUDA code.
   * Will be 0 when debug_mode is active.
   */
  int optimization_level{1};

  GeneratedCodeGen(const FunctionTrace<BaseScalar> &main_trace)
      : name_(main_trace.name), main_trace_(main_trace) {
    output_dim_ = main_trace_.output_dim;
    local_input_dim_ = main_trace_.input_dim;
  }

  GeneratedCodeGen(const std::string &name, std::shared_ptr<ADFun> tape)
      : name_(name) {
    main_trace_.tape = tape;
    output_dim_ = static_cast<int>(tape->Range());
    local_input_dim_ = static_cast<int>(tape->Domain());
    std::cout << "tape->Range():  " << tape->Range() << std::endl;
    std::cout << "tape->Domain(): " << tape->Domain() << std::endl;
  }

  // discards the compiled library (so that it gets recompiled at the next
  // evaluation)
  void discard_library() { library_name_ = ""; }

  const std::string &library_name() const { return library_name_; }
  void load_precompiled_library(const std::string &library_name) {
    if (library_name != library_name_) {
      discard_library();
    }
    library_name_ = library_name;
  }

  void set_global_input_dim(int dim) override {
    global_input_dim_ = dim;
    local_input_dim_ = main_trace_.tape->Domain() - dim;
    discard_library();
  }

  bool is_compiled() const { return !library_name_.empty(); }

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) override {
    if (target_ == TARGET_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      GenericModelPtr &model = get_cpu_model();
      model->ForwardZero(input, output);
#endif
    } else if (target_ == TARGET_CUDA) {
      const auto &model = get_cuda_model();
      model.forward_zero(input, output);
    }
  }

  void operator()(const std::vector<std::vector<BaseScalar>> &local_inputs,
                  std::vector<std::vector<BaseScalar>> &outputs,
                  const std::vector<BaseScalar> &global_input) override {
    outputs.resize(local_inputs.size());
    if (target_ == TARGET_CPU) {
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
    } else if (target_ == TARGET_CUDA) {
      const auto &model = get_cuda_model();
      model.forward_zero(&outputs, local_inputs, num_gpu_threads_per_block,
                         global_input);
    }
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override {
    if (target_ == TARGET_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      GenericModelPtr &model = get_cpu_model();
      model->Jacobian(input, output);
#endif
    } else if (target_ == TARGET_CUDA) {
      const auto &model = get_cuda_model();
      model.jacobian(input, output);
    }
  }

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input) override {
    outputs.resize(local_inputs.size());
    if (target_ == TARGET_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      for (auto &o : outputs) {
        o.resize(input_dim() * output_dim_);
      }
      int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
      for (int i = 0; i < num_tasks; ++i) {
        if (global_input.empty()) {
          GenericModelPtr &model = get_cpu_model();
          // model->ForwardZero(local_inputs[i], outputs[i]);
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
    } else if (target_ == TARGET_CUDA) {
      const auto &model = get_cuda_model();
      model.jacobian(&outputs, local_inputs, num_gpu_threads_per_block,
                     global_input);
    }
  }

  void compile_cpu() {
    using namespace CppAD;
    using namespace CppAD::cg;

    ModelCSourceGen<BaseScalar> main_source_gen(*(main_trace_.tape), name_);
    main_source_gen.setCreateJacobian(true);
    ModelLibraryCSourceGen<BaseScalar> libcgen(main_source_gen);
    // reverse order of invocation to first generate code for innermost
    // functions
    const auto &order = *CodeGenData<BaseScalar>::invocation_order;
    std::list<ModelCSourceGen<BaseScalar> *> models;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      FunctionTrace<BaseScalar> &trace = (*CodeGenData<BaseScalar>::traces)[*it];
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
    psave.saveSourcesTo(name_ + "_cpu_srcs");
    std::cerr << "CPU code compilation is not yet available on Windows. Saved "
                 "source files to \""
              << name_ << "_cpu_srcs"
              << "\".\n ";
    std::exit(1);
#else
    DynamicModelLibraryProcessor<BaseScalar> p(libcgen);
    auto compiler = std::make_unique<ClangCompiler<BaseScalar>>();
    compiler->setSourcesFolder(name_ + "_cpu_srcs");
    compiler->setSaveToDiskFirst(true);
    if (debug_mode) {
      compiler->addCompileFlag("-g");
      compiler->addCompileFlag("-O0");
    } else {
      compiler->addCompileFlag("-O" + std::to_string(optimization_level));
    }
    p.setLibraryName(name_ + "_cpu");
    p.createDynamicLibrary(*compiler, false);

    library_name_ = "./" + name_ + "_cpu.so";
#endif
    target_ = TARGET_CPU;
  }

#if !CPPAD_CG_SYSTEM_WIN
  mutable std::mutex cpu_library_loading_mutex_{};

  GenericModelPtr &get_cpu_model() const {
    if (!cpu_library_) {
      cpu_library_loading_mutex_.lock();
      cpu_library_ = std::make_shared<CppAD::cg::LinuxDynamicLib<BaseScalar>>(
          library_name_);
      // load and wire up atomic functions in this library
      const auto &order = *CodeGenData<BaseScalar>::invocation_order;
      const auto &hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
      cpu_models_[name_] =
          GenericModelPtr(cpu_library_->model(name_).release());
      for (const std::string &model_name : order) {
        // we have to keep the atomic function pointers alive
        cpu_models_[model_name] =
            GenericModelPtr(cpu_library_->model(model_name).release());
        // simply add every atomic to the top-level function
        // (because we don't have hierarchy information for the top-level
        // function)
        cpu_models_[name_]->addAtomicFunction(
            cpu_models_[model_name]->asAtomic());
      }
      for (const auto &[outer, inner_models] : hierarchy) {
        auto &outer_model = cpu_models_[outer];
        for (const std::string &inner : inner_models) {
          auto &inner_model = cpu_models_[inner];
          outer_model->addAtomicFunction(inner_model->asAtomic());
          std::cout << "Connected atomic function \"" + inner +
                           "\" to its parent function \""
                    << outer << "\".\n";
        }
      }

      std::cout << "Loaded compiled model \"" << name_ << "\" from \""
                << library_name_ << "\".\n";
      cpu_library_loading_mutex_.unlock();
    }
    return cpu_models_[name_];
  }
#endif

  void compile_cuda() {
    using namespace CppAD;
    using namespace CppAD::cg;

    std::cout << "Compiling CUDA code...\n";
    
    std::cout << "Invocation order: ";
    for (const auto &s : *(CodeGenData<BaseScalar>::invocation_order)) {
      std::cout << s << " ";
    }
    std::cout << std::endl;

    CudaModelSourceGen<BaseScalar> main_source_gen(*(main_trace_.tape), name_);
    main_source_gen.setCreateJacobian(true);
    main_source_gen.global_input_dim() = global_input_dim_;
    main_source_gen.jacobian_acc_method() = jac_acc_method_;
    CudaLibraryProcessor<BaseScalar> cuda_proc(&main_source_gen,
                                               name_ + "_cuda");
    // reverse order of invocation to first generate code for innermost
    // functions
    const auto &order = *CodeGenData<BaseScalar>::invocation_order;
    std::list<CudaModelSourceGen<BaseScalar> *> models;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      std::cout << "Adding cuda model " << *it << "\n";
      FunctionTrace<BaseScalar> &trace = (*CodeGenData<BaseScalar>::traces)[*it];
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
    cuda_proc.optimization_level() = optimization_level;
    cuda_proc.create_library();

    library_name_ = name_ + "_cuda";

    for (auto *model : models) {
      delete model;
    }
    target_ = TARGET_CUDA;
  }

  const CudaModel<BaseScalar> &get_cuda_model() const {
    if (!cuda_library_) {
      cuda_library_ = std::make_shared<CudaLibrary<BaseScalar>>(library_name_);
    }
    return cuda_library_->get_model(name_);
  }
};
}  // namespace autogen