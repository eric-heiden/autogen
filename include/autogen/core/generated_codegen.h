#pragma once

// clang-format off
#include "types.hpp"
#include "target_type.hpp"
#include "codegen.hpp"
// clang-format on

namespace autogen {
class Target;

class GeneratedCodeGen : public GeneratedBase {
  template <template <typename> typename Functor>
  friend struct Generated;

 public:
  using CppADScalar = typename CppAD::AD<BaseScalar>;
  using CGScalar = typename CppAD::AD<CppAD::cg::CG<BaseScalar>>;
  using ADFun = typename FunctionTrace::ADFun;

 protected:
  using GeneratedBase::global_input_dim_;
  using GeneratedBase::local_input_dim_;
  using GeneratedBase::output_dim_;

  std::string name_;
  FunctionTrace main_trace_;

  mutable std::shared_ptr<Target> target_;

 public:
  /**
   * The accumulation method determines if and how the individual Jacobian
   * matrices computed by the threads should get accumulated.
   */
  AccumulationMethod jac_acc_method{ACCUMULATE_NONE};

  GeneratedCodeGen(const FunctionTrace &main_trace)
      : name_(main_trace.name), main_trace_(main_trace) {
    output_dim_ = main_trace_.output_dim;
    local_input_dim_ = main_trace_.input_dim;
  }

  GeneratedCodeGen(const std::string &name, ADFun *tape) : name_(name) {
    main_trace_.tape = tape;
    output_dim_ = static_cast<int>(tape->Range());
    local_input_dim_ = static_cast<int>(tape->Domain());
    std::cout << "tape->Range():  " << tape->Range() << std::endl;
    std::cout << "tape->Domain(): " << tape->Domain() << std::endl;
  }

  GeneratedCodeGen(const std::string &name, std::shared_ptr<ADFun> tape)
      : name_(name) {
    main_trace_.tape = tape.get();
    output_dim_ = static_cast<int>(tape->Range());
    local_input_dim_ = static_cast<int>(tape->Domain());
    std::cout << "tape->Range():  " << tape->Range() << std::endl;
    std::cout << "tape->Domain(): " << tape->Domain() << std::endl;
  }

  const std::string &name() const { return name_; }

  FunctionTrace &main_trace() { return main_trace_; }

  TargetType target_type() const;
  std::shared_ptr<Target> target() const { return target_; }
  void set_target(std::shared_ptr<Target> target) { target_ = target; }
  void set_target(TargetType type);
  bool has_target() const { return target_ != nullptr; }

  void load_library(TargetType type, const std::string &filename);

  // discards the compiled library (so that it gets recompiled at the next
  // evaluation)
  void discard_library();

  const std::string &library_name() const;

  void set_global_input_dim(int dim) override {
    global_input_dim_ = dim;
    local_input_dim_ = static_cast<int>(main_trace_.tape->Domain() - dim);
    discard_library();
  }

  bool is_compiled() const;

  bool generate_code();
  bool compile();

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) override;

  void operator()(const std::vector<std::vector<BaseScalar>> &local_inputs,
                  std::vector<std::vector<BaseScalar>> &outputs,
                  const std::vector<BaseScalar> &global_input) override;

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override;

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input) override;
};
}  // namespace autogen
