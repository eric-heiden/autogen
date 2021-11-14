#include "autogen/core/generated_codegen.h"

#include "autogen/cg/target/cuda_target.h"
#include "autogen/cg/target/legacy_c_target.h"
#include "autogen/cg/target/openmp_target.h"

namespace autogen {
TargetType GeneratedCodeGen::target_type() const { return target_->type(); }

void GeneratedCodeGen::set_target(TargetType type) {
  switch (type) {
    case TARGET_LEGACY_C:
      target_ = std::make_shared<LegacyCTarget>(
          std::shared_ptr<GeneratedCodeGen>(this));
      break;
    case TARGET_CUDA:
      target_ =
          std::make_shared<CudaTarget>(std::shared_ptr<GeneratedCodeGen>(this));
      break;
    case TARGET_OPENMP:
      target_ =
          std::make_shared<OpenMpTarget>(std::shared_ptr<GeneratedCodeGen>(this));
      break;
    default:
      throw std::runtime_error("Unsupported target type");
  }
}

void GeneratedCodeGen::load_library(TargetType type,
                                    const std::string &filename) {
  set_target(type);
  target_->load_library(filename);
}

void GeneratedCodeGen::discard_library() {
  if (target_ != nullptr) {
    target_->discard_library();
  }
}

const std::string &GeneratedCodeGen::library_name() const {
  if (target_ != nullptr) {
    return target_->library_name();
  }
  static const std::string empty_string{""};
  return empty_string;
}

bool GeneratedCodeGen::is_compiled() const {
  return target_ != nullptr && target_->is_compiled();
}

void GeneratedCodeGen::operator()(const std::vector<BaseScalar> &input,
                                  std::vector<BaseScalar> &output) {
  target_->forward(input, output);
}

void GeneratedCodeGen::operator()(
    const std::vector<std::vector<BaseScalar>> &local_inputs,
    std::vector<std::vector<BaseScalar>> &outputs,
    const std::vector<BaseScalar> &global_input) {
  target_->forward(local_inputs, outputs, global_input);
}

void GeneratedCodeGen::jacobian(const std::vector<BaseScalar> &input,
                                std::vector<BaseScalar> &output) {
  target_->jacobian(input, output);
}

void GeneratedCodeGen::jacobian(
    const std::vector<std::vector<BaseScalar>> &local_inputs,
    std::vector<std::vector<BaseScalar>> &outputs,
    const std::vector<BaseScalar> &global_input) {
  target_->jacobian(local_inputs, outputs, global_input);
}

bool GeneratedCodeGen::generate_code() {
  if (!main_trace_.tape) {
    throw std::runtime_error(
        "No function trace is available to GeneratedCodeGen");
  }
  if (!target_) {
    throw std::runtime_error("No target has been assigned to GeneratedCodeGen");
  }
  return target_->generate_code();
}

bool GeneratedCodeGen::compile() {
  if (is_compiled()) {
    return true;
  }
  if (!main_trace_.tape) {
    throw std::runtime_error(
        "No function trace is available to GeneratedCodeGen");
  }
  if (!target_) {
    throw std::runtime_error("No target has been assigned to GeneratedCodeGen");
  }
  return target_->compile();
}
}  // namespace autogen