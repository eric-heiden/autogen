#pragma once

#include <cassert>
#include <mutex>

#include "autogen/core/base.hpp"
#include "autogen/core/codegen.hpp"
#include "autogen/core/target.hpp"

namespace autogen {
struct LegacyCTarget : public Target {
  using CGAtomicFunBridge =
      typename FunctionTrace::CGAtomicFunBridge;

  typedef CppAD::cg::GenericModel<BaseScalar> GenericModel;
  typedef std::shared_ptr<GenericModel> GenericModelPtr;

#if AUTOGEN_SYSTEM_WIN
  typedef CppAD::cg::WindowsDynamicLib<BaseScalar> DynamicLib;
#else
  typedef CppAD::cg::LinuxDynamicLib<BaseScalar> DynamicLib;
#endif

  using AbstractCCompiler = typename CppAD::cg::AbstractCCompiler<BaseScalar>;
  using MsvcCompiler = typename CppAD::cg::MsvcCompiler<BaseScalar>;
  using ClangCompiler = typename CppAD::cg::ClangCompiler<BaseScalar>;
  using GccCompiler = typename CppAD::cg::GccCompiler<BaseScalar>;

 protected:
  mutable std::shared_ptr<DynamicLib> cpu_library_{nullptr};
  mutable std::map<std::string, GenericModelPtr> cpu_models_;
  mutable std::mutex cpu_library_loading_mutex_{};
  mutable std::shared_ptr<AbstractCCompiler> cpu_compiler_{nullptr};
  using Target::cg_;
  using Target::sources_;
  using Target::sources_folder_;
  using Target::temp_folder_;
  using Target::library_name_;

#if AUTOGEN_SYSTEM_WIN
  std::string library_ext_{".dll"};
#else
  std::string library_ext_{".so"};
#endif

 private:
  std::shared_ptr<CppAD::cg::ModelLibraryCSourceGen<BaseScalar>> libcgen_{
      nullptr};

 public:
  LegacyCTarget(std::shared_ptr<GeneratedCodeGen> cg)
      : Target(cg, TARGET_LEGACY_C) {}

  void forward(const std::vector<BaseScalar> &input,
               std::vector<BaseScalar> &output) override;

  void forward(const std::vector<std::vector<BaseScalar>> &local_inputs,
               std::vector<std::vector<BaseScalar>> &outputs,
               const std::vector<BaseScalar> &global_input) override;

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override;

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input) override;

  GenericModelPtr get_cpu_model() const;

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

protected:
  bool generate_code_() override;

  bool compile_() override;
};
}  // namespace autogen
