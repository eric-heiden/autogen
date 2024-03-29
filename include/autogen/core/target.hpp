#pragma once

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "autogen/utils/cache.h"
#include "autogen/utils/filesystem.hpp"
#include "base.hpp"
#include "generated_codegen.h"

namespace autogen {

class Target {
 protected:
  // this GeneratedCodeGen instance is owned by Generated, not the Target
  GeneratedCodeGen *cg_{nullptr};
  TargetType type_;
  std::string name_;

  // mapping from file name to source code content
  std::vector<std::pair<std::string, std::string>> sources_;

  // file names of the source files which are compiled
  std::vector<std::string> source_filenames_;

  std::string library_name_;

  bool code_can_be_compiled_{true};

  /**
   * Whether the generated code is compiled in debug mode (only applies to CPU
   * and CUDA).
   */
  bool debug_mode_{false};

  /**
   * Optimization level to use when compiling the generated code.
   * Will be 0 when debug_mode is active.
   */
  int optimization_level_{2};

  /**
   * Whether to generate code for the zero-order forward mode.
   */
  bool generate_forward_{true};

  /**
   * Whether to generate code for the Jacobian.
   */
  bool generate_jacobian_{true};

  bool found_in_cache_{false};

  Target(GeneratedCodeGen *cg, TargetType type)
      : cg_(cg), type_(type), name_(cg->name()) {
    if (cg_ == nullptr) {
      throw std::runtime_error(
          "Target cannot be created from an empty pointer to GeneratedCodeGen");
    }
  }

  // some helper function to ease access to important info from GeneratedCodeGen
  // to the Target implementations

  int local_input_dim() const { return cg_->local_input_dim(); }
  int global_input_dim() const { return cg_->global_input_dim(); }
  int input_dim() const { return cg_->input_dim(); }
  int output_dim() const { return cg_->output_dim(); }

  const std::string &name() const { return name_; }

  FunctionTrace &main_trace() { return cg_->main_trace(); }

  virtual bool generate_code_() = 0;

  virtual bool compile_() {
    throw std::runtime_error(
        "function \"compile\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

 public:
  TargetType type() const { return type_; }

  virtual ~Target() = default;

  const std::string &library_name() const { return library_name_; }

  /**
   * Whether the code can be compiled via a compiler accessible through autogen,
   * or if an external tool is necessary to consume the generated code.
   */
  bool code_can_be_compiled() const { return code_can_be_compiled_; }
  virtual bool is_compiled() const { return !library_name_.empty(); }

  bool debug_mode() const { return debug_mode_; }
  void set_debug_mode(bool debug_mode) { debug_mode_ = debug_mode; }

  int optimization_level() const { return optimization_level_; }
  void set_optimization_level(int optimization_level) {
    optimization_level_ = optimization_level;
  }

  const std::vector<std::pair<std::string, std::string>> &sources() const {
    return sources_;
  }

  bool generate_code() {
    sources_.clear();
    source_filenames_.clear();
    if (!generate_code_()) {
      return false;
    }
    save_sources_();
    return true;
  }

  /**
   * Saves the generated source files to the folder defined by `src_dir()`.
   */
  virtual void save_sources_() {
    namespace fs = std::filesystem;
    if (sources_.empty()) {
      throw std::runtime_error(
          "No source files have been generated yet. Ensure "
          "`Target::generate_code()` is called before saving the code.");
    }
    found_in_cache_ = Cache::exists(sources_, &name_);
    if (found_in_cache_) {
      std::cout << "Sources for " << name() << " with target "
                << std::to_string(type_)
                << " have already been compiled. Loading library from cache."
                << std::endl;

    } else {
      Cache::save_sources(sources_, type_, name());
    }
  }

  bool compile() {
    library_name_ = Cache::get_library_file(this->type(), this->name());
    if (!found_in_cache_) {
      autogen::load_windows_build_variables();
      if (!compile_()) {
        return false;
      }
    }
    load_library(library_name_);
    return true;
  }

  virtual bool load_library(const std::string &filename) {
    throw std::runtime_error(
        "function \"load_library\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }
  virtual bool discard_library() {
    throw std::runtime_error(
        "function \"discard_library\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

  virtual void forward(const std::vector<BaseScalar> &input,
                       std::vector<BaseScalar> &output) {
    throw std::runtime_error(
        "function \"forward\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

  virtual void forward(const std::vector<std::vector<BaseScalar>> &local_inputs,
                       std::vector<std::vector<BaseScalar>> &outputs,
                       const std::vector<BaseScalar> &global_input) {
    throw std::runtime_error(
        "function \"forward\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

  virtual void jacobian(const std::vector<BaseScalar> &input,
                        std::vector<BaseScalar> &output) {
    throw std::runtime_error(
        "function \"jacobian\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

  virtual void jacobian(
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      std::vector<std::vector<BaseScalar>> &outputs,
      const std::vector<BaseScalar> &global_input) {
    throw std::runtime_error(
        "function \"jacobian\" has not been implemented for target " +
        std::to_string(type_) + ".");
  }

  virtual bool create_cmake_project(
      const std::string &destination_folder,
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      const std::vector<BaseScalar> &global_input) const {
    throw std::runtime_error(
        "function \"create_cmake_project\" has not been implemented for "
        "target" +
        std::to_string(type_) + ".");
  }

  virtual bool create_cmake_project(
      const std::string &destination_folder,
      const std::vector<std::vector<BaseScalar>> &local_inputs) {
    return create_cmake_project(
        destination_folder, local_inputs,
        std::vector<BaseScalar>(global_input_dim(), 0.0));
  }

  virtual bool create_cmake_project(const std::string &destination_folder,
                                    const std::vector<BaseScalar> &input) {
    std::vector<std::vector<BaseScalar>> inputs{input};
    return create_cmake_project(destination_folder, inputs);
  }

  bool create_cmake_project(
      const std::vector<BaseScalar> &input,
      const std::vector<BaseScalar> &global_input = std::vector<BaseScalar>{}) {
    std::vector<std::vector<BaseScalar>> inputs{input};
    std::string folder_name =
        Cache::get_cmake_folder(this->type(), this->name());
    return create_cmake_project(folder_name, inputs, global_input);
  }
};
}  // namespace autogen