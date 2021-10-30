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

  using AbstractCCompiler = typename CppAD::cg::AbstractCCompiler<double>;
  using MsvcCompiler = typename CppAD::cg::MsvcCompiler<double>;
  using ClangCompiler = typename CppAD::cg::ClangCompiler<double>;
  using GccCompiler = typename CppAD::cg::GccCompiler<double>;

 private:
  using CGAtomicFunBridge =
      typename FunctionTrace<BaseScalar>::CGAtomicFunBridge;

  typedef CppAD::cg::GenericModel<BaseScalar> GenericModel;
  typedef std::shared_ptr<GenericModel> GenericModelPtr;

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

#if AUTOGEN_SYSTEM_WIN
  typedef CppAD::cg::WindowsDynamicLib<BaseScalar> DynamicLib;
#else
  typedef CppAD::cg::PosixDynamicLib<BaseScalar> DynamicLib;
#endif
  mutable std::shared_ptr<DynamicLib> cpu_library_{nullptr};
  mutable std::map<std::string, GenericModelPtr> cpu_models_;

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
  int optimization_level{2};

  /**
   * Whether to generate code for the zero-order forward mode.
   */
  bool generate_forward{true};

  /**
   * Whether to generate code for the Jacobian.
   */
  bool generate_jacobian{true};

  CodeGenTarget target() const { return target_; }
  void set_target(CodeGenTarget target) { target_ = target; }

  std::shared_ptr<AbstractCCompiler> cpu_compiler{nullptr};

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

  void set_cpu_compiler_clang(
      std::string compiler_path = "",
      const std::vector<std::string> &compile_flags =
          std::vector<std::string>{},
      const std::vector<std::string> &compile_lib_flags = {}) {
    if (compiler_path.empty()) {
      compiler_path = autogen::find_exe("clang");
    }
    cpu_compiler = std::make_shared<ClangCompiler>(compiler_path);
    for (const auto &flag : compile_flags) {
      cpu_compiler->addCompileFlag(flag);
    }
    for (const auto &flag : compile_lib_flags) {
      cpu_compiler->addCompileLibFlag(flag);
    }
  }

  void set_cpu_compiler_gcc(
      std::string compiler_path = "",
      const std::vector<std::string> &compile_flags =
          std::vector<std::string>{},
      const std::vector<std::string> &compile_lib_flags = {}) {
    if (compiler_path.empty()) {
      compiler_path = autogen::find_exe("gcc");
    }
    cpu_compiler = std::make_shared<GccCompiler>(compiler_path);
    for (const auto &flag : compile_flags) {
      cpu_compiler->addCompileFlag(flag);
    }
    for (const auto &flag : compile_lib_flags) {
      cpu_compiler->addCompileLibFlag(flag);
    }
  }

  void set_cpu_compiler_msvc(
      std::string compiler_path = "", std::string linker_path = "",
      const std::vector<std::string> &compile_flags =
          std::vector<std::string>{},
      const std::vector<std::string> &compile_lib_flags = {}) {
    if (compiler_path.empty()) {
      compiler_path = autogen::find_exe("cl.exe");
    }
    if (linker_path.empty()) {
      linker_path = autogen::find_exe("link.exe");
    }
    cpu_compiler = std::make_shared<MsvcCompiler>(compiler_path, linker_path);
    for (const auto &flag : compile_flags) {
      cpu_compiler->addCompileFlag(flag);
    }
    for (const auto &flag : compile_lib_flags) {
      cpu_compiler->addCompileLibFlag(flag);
    }
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
    local_input_dim_ = static_cast<int>(main_trace_.tape->Domain() - dim);
    discard_library();
  }

  bool is_compiled() const { return !library_name_.empty(); }

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) override {
    if (target_ == TARGET_CPU) {
      assert(!library_name_.empty());
      auto model = get_cpu_model();
      model->ForwardZero(input, output);
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
      assert(!library_name_.empty());
      for (auto &o : outputs) {
        o.resize(output_dim_);
      }
      int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
      for (int i = 0; i < num_tasks; ++i) {
        if (global_input.empty()) {
          auto model = get_cpu_model();
          model->ForwardZero(local_inputs[i], outputs[i]);
        } else {
          static thread_local std::vector<BaseScalar> input;
          input = global_input;
          input.resize(global_input.size() + local_inputs[0].size());
          for (size_t j = 0; j < local_inputs[i].size(); ++j) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          auto model = get_cpu_model();
          model->ForwardZero(input, outputs[i]);
        }
      }
    } else if (target_ == TARGET_CUDA) {
      const auto &model = get_cuda_model();
      model.forward_zero(&outputs, local_inputs, num_gpu_threads_per_block,
                         global_input);
    }
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override {
    if (target_ == TARGET_CPU) {
      assert(!library_name_.empty());
      auto model = get_cpu_model();
      model->Jacobian(input, output);
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
      assert(!library_name_.empty());
      for (auto &o : outputs) {
        o.resize(input_dim() * output_dim_);
      }
      int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
      for (int i = 0; i < num_tasks; ++i) {
        if (global_input.empty()) {
          auto model = get_cpu_model();
          // model->ForwardZero(local_inputs[i], outputs[i]);
          model->Jacobian(local_inputs[i], outputs[i]);
        } else {
          static thread_local std::vector<BaseScalar> input;
          if (input.empty()) {
            input.resize(global_input.size());
            input.insert(input.begin(), global_input.begin(),
                         global_input.end());
          }
          for (size_t j = 0; j < local_inputs[i].size(); ++j) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          auto model = get_cpu_model();
          model->Jacobian(input, outputs[i]);
        }
      }
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
    main_source_gen.setCreateForwardZero(generate_forward);
    main_source_gen.setCreateJacobian(generate_jacobian);
    ModelLibraryCSourceGen<BaseScalar> libcgen(main_source_gen);
    // reverse order of invocation to first generate code for innermost
    // functions
    const auto &order = *CodeGenData<BaseScalar>::invocation_order;
    std::list<ModelCSourceGen<BaseScalar> *> models;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      FunctionTrace<BaseScalar> &trace =
          (*CodeGenData<BaseScalar>::traces)[*it];
      // trace.tape->optimize();
      auto *source_gen = new ModelCSourceGen<BaseScalar>(*(trace.tape), *it);
      source_gen->setCreateForwardZero(generate_forward);
      // source_gen->setCreateSparseJacobian(generate_jacobian);
      // source_gen->setCreateJacobian(generate_jacobian);
      source_gen->setCreateForwardOne(generate_jacobian);
      source_gen->setCreateReverseOne(generate_jacobian);
      models.push_back(source_gen);
      // we need a stable reference
      libcgen.addModel(*(models.back()));
    }
    libcgen.setVerbose(true);

    DynamicModelLibraryProcessor<BaseScalar> p(libcgen);

    // if (clang_path.empty()) {
    //   clang_path = autogen::find_exe("clang", false);
    // }
    // if (clang_path.empty()) {
    //   throw std::runtime_error(
    //       "Clang path is empty, make sure clang is "
    //       "available on the system path or provide it manually to the "
    //       "GeneratedCodegen instance.");
    // }
    // auto compiler = std::make_unique<ClangCompiler<BaseScalar>>(clang_path);
    if (!cpu_compiler) {
#if AUTOGEN_SYSTEM_WIN
      set_cpu_compiler_msvc();
#else
      set_cpu_compiler_clang();
#endif
    }
    cpu_compiler->setSourcesFolder(name_ + "_cpu_srcs");
    cpu_compiler->setTemporaryFolder(name_ + "_cpu_tmp");
    cpu_compiler->setSaveToDiskFirst(true);
    if (debug_mode) {
      cpu_compiler->addCompileFlag("-g");
      cpu_compiler->addCompileFlag("-O0");
    } else {
      cpu_compiler->addCompileFlag("-O" + std::to_string(optimization_level));
    }
    p.setLibraryName(name_ + "_cpu");
    bool load_library = false;  // we do this in another step
    p.createDynamicLibrary(*cpu_compiler, load_library);
    library_name_ = "./" + name_ + "_cpu";
    target_ = TARGET_CPU;
  }

  mutable std::mutex cpu_library_loading_mutex_{};

  GenericModelPtr get_cpu_model() const {
    if (!cpu_library_) {
      cpu_library_loading_mutex_.lock();
      cpu_library_ = std::make_shared<DynamicLib>(library_name_ + library_ext_);
      std::set<std::string> model_names = cpu_library_->getModelNames();
      std::cout << "Successfully loaded CPU library "
                << library_name_ + library_ext_ << std::endl;
      for (auto &name : model_names) {
        std::cout << "  Found model " << name << std::endl;
      }
      // load and wire up atomic functions in this library
      const auto &order = *CodeGenData<BaseScalar>::invocation_order;
      const auto &hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
      cpu_models_[name_] =
          GenericModelPtr(cpu_library_->model(name_).release());
      if (!cpu_models_[name_]) {
        throw std::runtime_error("Failed to load model from library " +
                                 library_name_ + library_ext_);
      }
      // atomic functions to be added
      typedef std::pair<std::string, std::string> ParentChild;
      std::set<ParentChild> remaining_atomics;
      for (const std::string &s :
           cpu_models_[name_]->getAtomicFunctionNames()) {
        remaining_atomics.insert(std::make_pair(name_, s));
      }
      while (!remaining_atomics.empty()) {
        ParentChild member = *(remaining_atomics.begin());
        const std::string &parent = member.first;
        const std::string &atomic_name = member.second;
        remaining_atomics.erase(remaining_atomics.begin());
        if (cpu_models_.find(atomic_name) == cpu_models_.end()) {
          std::cout << "  Adding atomic function " << atomic_name << std::endl;
          cpu_models_[atomic_name] =
              GenericModelPtr(cpu_library_->model(atomic_name).release());
          for (const std::string &s :
               cpu_models_[atomic_name]->getAtomicFunctionNames()) {
            remaining_atomics.insert(std::make_pair(atomic_name, s));
          }
        }
        auto &atomic_model = cpu_models_[atomic_name];
        cpu_models_[parent]->addAtomicFunction(atomic_model->asAtomic());
      }

      std::cout << "Loaded compiled model \"" << name_ << "\" from \""
                << library_name_ << "\".\n";
      cpu_library_loading_mutex_.unlock();
    }
    return cpu_models_[name_];
  }

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
    main_source_gen.setCreateForwardZero(generate_forward);
    main_source_gen.setCreateJacobian(generate_jacobian);
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
      FunctionTrace<BaseScalar> &trace =
          (*CodeGenData<BaseScalar>::traces)[*it];
      auto *source_gen = new CudaModelSourceGen<BaseScalar>(*(trace.tape), *it);
      source_gen->setCreateForwardOne(generate_jacobian);
      source_gen->setCreateReverseOne(generate_jacobian);
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

 private:
#if AUTOGEN_SYSTEM_WIN
  static const inline std::string library_ext_ = ".dll";
#else
  static const inline std::string library_ext_ = ".so";
#endif
};
}  // namespace autogen