#pragma once

#include <cassert>
#include <mutex>

#include "autogen/core/base.hpp"
#include "autogen/core/codegen.hpp"
#include "autogen/core/target.hpp"

namespace autogen {
struct LegacyCTarget : public Target {
  using CGAtomicFunBridge = typename FunctionTrace::CGAtomicFunBridge;

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
  std::shared_ptr<DynamicLib> cpu_library_{nullptr};
  std::map<std::string, GenericModelPtr> lib_models_;
  GenericModelPtr main_model_{nullptr};
  std::mutex cpu_library_loading_mutex_{};
  std::shared_ptr<AbstractCCompiler> compiler_{nullptr};

  using Target::cg_;
  using Target::library_name_;
  using Target::sources_;
  using Target::sources_folder_;
  using Target::temp_folder_;

#if AUTOGEN_SYSTEM_WIN
  std::string library_ext_{".dll"};
#else
  std::string library_ext_{".so"};
#endif

 private:
  std::shared_ptr<CppAD::cg::ModelLibraryCSourceGen<BaseScalar>> libcgen_{
      nullptr};

  std::list<CppAD::cg::ModelCSourceGen<BaseScalar> *> models_;

 public:
  LegacyCTarget(GeneratedCodeGen *cg) : Target(cg, TARGET_LEGACY_C) {}

  virtual ~LegacyCTarget() {
    for (auto *model : models_) {
      delete model;
    }
  }

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

  bool load_library(const std::string &filename) override;

 protected:
  bool generate_code_() override;

  bool compile_() override;
};
}  // namespace autogen
