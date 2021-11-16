#pragma once

#include "autogen/cg/compact/compact_codegen.h"
#include "autogen/cg/compact/compact_library.hpp"
#include "autogen/cg/compact/compact_model.hpp"
#include "autogen/core/target.hpp"
#include "autogen/utils/file_utils.hpp"

namespace autogen {
/**
 * Abstract Target class that implements helpers to generate the code for the
 * traced function.
 */
template <class CodeGenT = CompactCodeGen,
          class LibFunctionT = CompactLibFunction>
struct CompactTarget : public Target {
  using LibFunction = LibFunctionT;
  using CodeGen = CodeGenT;

 protected:
  using Target::cg_;
  using Target::debug_mode_;
  using Target::sources_;
  using Target::sources_folder_;
  using Target::type_;

  std::unique_ptr<CodeGen> codegen_{nullptr};
  std::unique_ptr<CompactLibrary<LibFunction>> library_{nullptr};

  // weak pointer to the model of interest in the library
  CompactModel<LibFunction> *model_{nullptr};

  CompactTarget(GeneratedCodeGen *cg, TargetType type) : Target(cg, type) {
    if (!main_trace().has_tape()) {
      throw std::runtime_error(
          "CompactTarget cannot be created until GeneratedCodeGen has traced "
          "the function.");
    }
    bool is_function_only = false;
    codegen_ = std::make_unique<CodeGen>(name(), is_function_only);
    codegen_->set_tape(*main_trace().tape);
  }

  virtual bool generate_code_();

  /**
   * Saves the generated source files to the folder defined by `src_dir()`.
   */
  void save_sources_() const {
    namespace fs = std::filesystem;
    if (sources_.empty()) {
      throw std::runtime_error(
          "No source files have been generated yet. Ensure "
          "`Target::generate_code()` is called before saving the code.");
    }
    fs::create_directories(sources_folder_);
    std::cout << "Saving source files at "
              << FileUtils::abs_path(sources_folder_) << "\n";
    for (const auto &entry : sources_) {
      std::ofstream file(fs::path(sources_folder_) / entry.first);
      file << entry.second;
      file.close();
    }
  }

  virtual std::string util_header_src_() const;
  virtual std::string model_info_src_() const;

 public:
  using Target::global_input_dim;
  using Target::input_dim;
  using Target::local_input_dim;
  using Target::main_trace;
  using Target::name;
  using Target::output_dim;

  virtual ~CompactTarget() { discard_library(); }

  virtual bool is_compiled() const { return model_ != nullptr; }

  bool load_library(const std::string &filename) override {
    library_ = std::make_unique<CompactLibrary<LibFunction>>();
    library_->load(filename);
    if (!library_->has_model(name())) {
      throw std::runtime_error("The library \"" + filename +
                               "\" does not contain the model \"" + name() +
                               "\".");
    }
    model_ = library_->get_model(name());
    return true;
  }

  bool discard_library() override {
    if (model_ != nullptr) {
      library_.reset();
      model_ = nullptr;  // this weak pointer is no longer valid
    }
    return true;
  }

  void forward(const std::vector<BaseScalar> &input,
               std::vector<BaseScalar> &output) override {
    output.resize(output_dim());
    model_->forward_zero(input, output);
  }

  void forward(const std::vector<std::vector<BaseScalar>> &local_inputs,
               std::vector<std::vector<BaseScalar>> &outputs,
               const std::vector<BaseScalar> &global_input) override {
    outputs.resize(local_inputs.size());
    for (auto &o : outputs) {
      o.resize(output_dim());
    }
    model_->forward_zero(&outputs, local_inputs, global_input);
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override {
    const auto jac_dim = input_dim() * output_dim();
    output.resize(jac_dim);
    model_->jacobian(input, output);
  }

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input = {}) override {
    const auto jac_dim = input_dim() * output_dim();
    outputs.resize(local_inputs.size());
    for (auto &o : outputs) {
      o.resize(jac_dim);
    }
    model_->jacobian(&outputs, local_inputs, global_input);
  }

  virtual bool create_cmake_project(
      const std::string &destination_folder,
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      const std::vector<BaseScalar> &global_input) const;
};
}  // namespace autogen
