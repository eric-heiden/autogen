#pragma once

#include "autogen/cg/compact/compact_codegen.h"
#include "autogen/core/target.hpp"

namespace autogen {

class OpenMpCodeGen : public CompactCodeGen {
 protected:
  using CompactCodeGen::global_input_dim;
  using CompactCodeGen::jac_acc_method_;
  using CompactCodeGen::local_input_dim;
  using CompactCodeGen::model_name_;
  using CompactCodeGen::output_dim;

 public:
  OpenMpCodeGen(const std::string &model_name, bool function_only = false)
      : CompactCodeGen(model_name, function_only) {}

  /**
   * Whether the kernel function should take the thread ID as an argument,
   * or whether it can be retrieved in some other way.
   */
  bool kernel_requires_id_argument() const override { return true; }

  void emit_kernel_launch(std::ostringstream &code,
                          const std::string &function_name,
                          size_t local_input_dim, size_t global_input_dim,
                          size_t output_dim) const override;

  void emit_allocation_functions(std::ostringstream &code,
                                 const std::string &function_name,
                                 size_t local_input_dim,
                                 size_t global_input_dim,
                                 size_t output_dim) const override;

  void emit_send_functions(std::ostringstream &code,
                           const std::string &function_name,
                           size_t local_input_dim, size_t global_input_dim,
                           size_t output_dim) const override;

  void emit_cmake_code(std::ostringstream &code,
                       const std::string &project_name) const override;
};

struct OpenMpTarget : public CompactTarget<OpenMpCodeGen> {
  using typename CompactTargetT = CompactTarget<OpenMpCodeGen>;

  using AbstractCCompiler = typename CppAD::cg::AbstractCCompiler<BaseScalar>;
  using MsvcCompiler = typename CppAD::cg::MsvcCompiler<BaseScalar>;
  using ClangCompiler = typename CppAD::cg::ClangCompiler<BaseScalar>;
  using GccCompiler = typename CppAD::cg::GccCompiler<BaseScalar>;

 protected:
  mutable std::shared_ptr<CompactLibrary<>> library_{nullptr};

  mutable std::shared_ptr<AbstractCCompiler> compiler_{nullptr};

 public:
  OpenMpTarget(std::shared_ptr<GeneratedCodeGen> cg)
      : CompactTargetT(cg, TargetType::TARGET_OPENMP) {}

 protected:
  bool compile_();

  void set_compiler_clang(
      std::string compiler_path = "",
      const std::vector<std::string> &compile_flags =
          std::vector<std::string>{},
      const std::vector<std::string> &compile_lib_flags = {});

  void set_compiler_gcc(std::string compiler_path = "",
                        const std::vector<std::string> &compile_flags =
                            std::vector<std::string>{},
                        const std::vector<std::string> &compile_lib_flags = {});

  void set_compiler_msvc(
      std::string compiler_path = "", std::string linker_path = "",
      const std::vector<std::string> &compile_flags =
          std::vector<std::string>{},
      const std::vector<std::string> &compile_lib_flags = {});
};
}  // namespace autogen
