#pragma once

// clang-format off
#include <functional>

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>

#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif

#include "base.hpp"
// clang-format on

namespace autogen {
class GeneratedCppAD : public GeneratedBase {
  using ADScalar = typename CppAD::AD<BaseScalar>;

 private:
  std::shared_ptr<CppAD::ADFun<BaseScalar>> tape_{nullptr};

  std::vector<ADScalar> ax_;
  std::vector<ADScalar> ay_;

 protected:
  using GeneratedBase::global_input_dim_;
  using GeneratedBase::local_input_dim_;
  using GeneratedBase::output_dim_;

 public:
  GeneratedCppAD(const std::vector<ADScalar>& ax,
                 const std::vector<ADScalar>& ay) {
    tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
    tape_->Dependent(ax, ay);
  }

  GeneratedCppAD(std::shared_ptr<CppAD::ADFun<BaseScalar>> tape)
      : tape_(tape) {}

  GeneratedCppAD(
      std::function<void(const std::vector<ADScalar>&, std::vector<ADScalar>&)>
          functor,
      const std::vector<BaseScalar>& input) {
    ax_.resize(input.size());
    // TODO resize ay_?
    for (size_t i = 0; i < input.size(); ++i) {
      ax_[i] = ADScalar(input[i]);
    }
    CppAD::Independent(ax_);
    functor(ax_, ay_);
    tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
    tape_->Dependent(ax_, ay_);
  }

  void operator()(const std::vector<BaseScalar>& input,
                  std::vector<BaseScalar>& output) override {
    output = tape_->Forward(0, input);
  }

  void operator()(const std::vector<std::vector<BaseScalar>>& local_inputs,
                  std::vector<std::vector<BaseScalar>>& outputs,
                  const std::vector<BaseScalar>& global_input) override {
    if (global_input.empty()) {
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        outputs[i] = tape_->Forward(0, local_inputs[i]);
      }
    } else {
      std::vector<BaseScalar> input(global_input);
      input.resize(global_input.size() + local_inputs[0].size());
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        for (size_t j = 0; j < local_inputs[i].size(); ++j) {
          input[j + global_input.size()] = local_inputs[i][j];
        }
        outputs[i] = tape_->Forward(0, input);
      }
    }
  }

  void jacobian(const std::vector<BaseScalar>& input,
                std::vector<BaseScalar>& output) override {
    output = tape_->Jacobian(input);
  }

  void jacobian(const std::vector<std::vector<BaseScalar>>& local_inputs,
                std::vector<std::vector<BaseScalar>>& outputs,
                const std::vector<BaseScalar>& global_input) override {
    outputs.resize(local_inputs.size());
    if (global_input.empty()) {
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        jacobian(local_inputs[i], outputs[i]);
      }
    } else {
      std::vector<BaseScalar> input(global_input.size());
      input.insert(input.begin(), global_input.begin(), global_input.end());
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        for (size_t j = 0; j < local_inputs[i].size(); ++i) {
          input[j + global_input.size()] = local_inputs[i][j];
        }
        jacobian(input, outputs[i]);
      }
    }
  }
};
}  // namespace autogen