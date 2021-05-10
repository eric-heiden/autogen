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
    ax.resize(input.size());
    ay.resize;
    for (size_t i = 0; i < input.size(); ++i) {
      ax[i] = ADScalar(input[i]);
    }
    CppAD::Independent(ax);
    functor(ax, ay);
    tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
    tape_->Dependent(ax, ay);
  }

  void operator()(const std::vector<BaseScalar>& input,
                  std::vector<BaseScalar>& output) override {}

 private:
  void conditionally_trace() {}
};
}  // namespace autogen