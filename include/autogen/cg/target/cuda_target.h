#pragma once

#include <filesystem>

#include "autogen/cg/compact/compact_codegen.h"
#include "autogen/cg/compact/compact_language.hpp"
#include "autogen/cg/compact/compact_library.hpp"
#include "autogen/cg/compiler/nvcc_compiler.h"
#include "autogen/core/target.hpp"
#include "autogen/utils/stopwatch.hpp"
#include "autogen/utils/system.h"

namespace autogen {

struct CudaLibFunctionTypes {
  using Scalar = BaseScalar;
  // signature: (num_threads, num_blocks, num_threads_per_block, output_vector)
  using LaunchFunctionPtrT = void (*)(int, int, int, BaseScalar *);
  using SendLocalFunctionPtrT = bool (*)(int, const BaseScalar *);
  using SendGlobalFunctionPtrT = bool (*)(const BaseScalar *);
  using MetaDataFunctionPtrT = CompactFunctionMetaData (*)();
  using AllocateFunctionPtrT = void (*)(int);
  using DeallocateFunctionPtrT = void (*)();
};

class CudaLibFunction : public CompactLibFunctionT<CudaLibFunctionTypes> {
 protected:
  using CompactLibFunctionT<CudaLibFunctionTypes>::function_name;
  using CompactLibFunctionT<CudaLibFunctionTypes>::fun_;
  using CompactLibFunctionT<CudaLibFunctionTypes>::is_available_;
  using CompactLibFunctionT<CudaLibFunctionTypes>::send_global_fun_;
  using CompactLibFunctionT<CudaLibFunctionTypes>::send_local_input;
  using CompactLibFunctionT<CudaLibFunctionTypes>::send_global_input;

  int num_threads_per_block_{128};  // TODO expose this in CudaTarget

  bool execute_(int num_threads, Scalar *output) const override {
    int num_threads_per_block = std::min(num_threads_per_block_, num_threads);
    int num_blocks = static_cast<int>(
        std::ceil(double(num_threads) / num_threads_per_block));
    fun_(num_threads, num_blocks, num_threads_per_block, output);
    return true;
  }
};

class CudaCodeGen : public CompactCodeGen {
 protected:
  using CompactCodeGen::global_input_dim;
  using CompactCodeGen::jac_acc_method_;
  using CompactCodeGen::local_input_dim;
  using CompactCodeGen::model_name_;
  using CompactCodeGen::output_dim;

 public:
  CudaCodeGen(const std::string &model_name, bool function_only = false,
              bool debug_mode = false)
      : CompactCodeGen(model_name, function_only, debug_mode) {
    this->kernel_requires_id_argument_ = false;
  }

  /**
   * Whether the kernel function should take the thread ID as an argument,
   * or whether it can be retrieved in some other way.
   */
  bool kernel_requires_id_argument() const override { return false; }
  /**
   * String to be prepended to the function signature.
   */
  std::string function_type_prefix(bool is_function) const override {
    return is_function ? "__device__\n" : "__global__\n";
  }

  std::string header_file_extension() const override { return ".cuh"; }
  std::string source_file_extension() const override { return ".cu"; }

  void emit_thread_id_getter(std::ostringstream &code) const override {
    code << "  const int ti = blockIdx.x * blockDim.x + threadIdx.x;\n";
  }

  void emit_allocation_functions(std::ostringstream &code,
                                 const std::string &function_name,
                                 size_t local_input_dim,
                                 size_t global_input_dim,
                                 size_t output_dim) const override;

  void emit_send_functions(std::ostringstream &code,
                           const std::string &function_name,
                           size_t local_input_dim, size_t global_input_dim,
                           size_t output_dim) const override;

  void emit_kernel_launch(std::ostringstream &code,
                          const std::string &function_name,
                          size_t local_input_dim, size_t global_input_dim,
                          size_t output_dim) const override;

  void emit_cmake_code(std::ostringstream &code,
                       const std::string &project_name) const override;

  void emit_cpp_function_call(std::ostringstream &code,
                              const std::string &fun_name) const override;
};

struct CudaTarget : public CompactTarget<CudaCodeGen, CudaLibFunction> {
  using CompactTargetT = CompactTarget<CudaCodeGen, CudaLibFunction>;

 protected:
  mutable std::shared_ptr<CompactLibrary<CudaLibFunctionTypes>> cuda_library_{
      nullptr};

  mutable std::shared_ptr<CppAD::cg::AbstractCCompiler<BaseScalar>> compiler_{
      nullptr};

  using CompactTargetT::codegen_;
  using Target::library_name_;
  using Target::optimization_level_;

 public:
  CudaTarget(GeneratedCodeGen *cg)
      : CompactTargetT(cg, TargetType::TARGET_CUDA) {}

  void set_compiler_nvcc(std::string compiler_path = "");

 protected:
  std::string util_header_src_() const override;

  // bool generate_code_() override {
  //   using namespace CppAD;
  //   using namespace CppAD::cg;

  //   std::cout << "Compiling CUDA code...\n";

  //   std::cout << "Invocation order: ";
  //   for (const auto &s : *(CodeGenData::invocation_order)) {
  //     std::cout << s << " ";
  //   }
  //   std::cout << std::endl;

  //   CompactModelSourceGen main_source_gen(*(main_trace_.tape), name_);
  //   main_source_gen.setCreateForwardZero(this->cg_->generate_forward);
  //   main_source_gen.setCreateJacobian(this->cg_->generate_jacobian);
  //   main_source_gen.global_input_dim() = this->global_input_dim();
  //   main_source_gen.jacobian_jac_acc_method_() =
  //   this->cg_->jac_jac_acc_method_; CudaLibraryProcessor<BaseScalar>
  //   cuda_proc(&main_source_gen,
  //                                              this->name() + "_cuda");
  //   // reverse order of invocation to first generate code for innermost
  //   // functions
  //   const auto &order = *CodeGenData::invocation_order;
  //   std::list<CompactModelSourceGen *> models;
  //   for (auto it = order.rbegin(); it != order.rend(); ++it) {
  //     std::cout << "Adding cuda model " << *it << "\n";
  //     FunctionTrace &trace = (*CodeGenData::traces)[*it];
  //     auto *source_gen = new CompactModelSourceGen(*(trace.tape), *it);
  //     source_gen->setCreateForwardOne(this->cg_->generate_jacobian);
  //     source_gen->setCreateReverseOne(this->cg_->generate_jacobian);
  //     source_gen->set_kernel_only(true);
  //     models.push_back(source_gen);
  //     cuda_proc.add_model(models.back(), false);
  //   }
  //   cuda_proc.debug_mode() = this->cg_->debug_mode;
  //   cuda_proc.generate_code();
  //   cuda_proc.save_sources();
  //   cuda_proc.optimization_level() = this->cg_->optimization_level;
  //   cuda_proc.create_library();

  //   for (auto *model : models) {
  //     delete model;
  //   }

  //   return true;
  // }

  /**
   * Compiles the previously generated code to a shared library file that can be
   * loaded subsequently.
   */
  bool compile_() override {
    if (compiler_ == nullptr) {
      set_compiler_nvcc();
    }
    std::stringstream cmd;
    std::string ncc_path = compiler_->getCompilerPath();
    std::cout << "Compiling CUDA library via " << ncc_path << std::endl;
    cmd << "\"" << ncc_path << "\" ";
    cmd << "--ptxas-options=-O" << std::to_string(optimization_level_)
        << ",-v ";
    cmd << "--ptxas-options=-v "
        << "-rdc=true ";
    // if (debug_mode_) {
    //   cmd << "-G ";
    // }
#if AUTOGEN_SYSTEM_WIN
    cmd << "-o " << library_name_ << ".dll "
#else
    cmd << "--compiler-options "
        << "-fPIC "
        << "-o " << library_name_ << ".so "
#endif
        << "--shared ";
    std::string sources_folder =
        Cache::get_source_folder(this->type(), this->name());
    std::filesystem::path folder(sources_folder);

    std::string project_name =
        Cache::get_project_name(this->type(), this->name());
    cmd << (folder / (project_name + ".cu")).string();
    autogen::Stopwatch timer;
    std::cout << "\n\n" << cmd.str() << "\n\n";
    timer.start();
    int return_code = std::system(cmd.str().c_str());
    timer.stop();
    std::cout << "CUDA compilation process terminated after " << timer.elapsed()
              << " seconds.\n";
    if (return_code) {
      throw std::runtime_error("CUDA compilation failed with return code " +
                               std::to_string(return_code) + ".");
    }
    return true;
  }
};
}  // namespace autogen