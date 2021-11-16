#pragma once

#include <numeric>

#include "autogen/core/types.hpp"
#include "compact_language.hpp"
#include "compact_variable_name_gen.hpp"

namespace autogen {
struct CompactModelSourceGen;

class CompactCodeGen {
  using CGBase = typename CppAD::cg::CG<BaseScalar>;

 protected:
  size_t global_input_dim_{0};

  AccumulationMethod jac_acc_method_{ACCUMULATE_MEAN};

  std::vector<size_t> jac_local_input_sparsity_;
  std::vector<size_t> jac_global_input_sparsity_;
  std::vector<size_t> jac_output_sparsity_;

  /**
   * Whether to only generate the function code, not the kernel launch and
   * memory access functions.
   */
  bool function_only_{false};

  bool debug_mode_{false};

  // TODO create Language just once if possible
  // LanguageCompact language_;

  // pointer to the gradient tape (which is not owned by this class)
  CppAD::ADFun<CGBase> *fun_{nullptr};
  std::string model_name_;

  mutable std::shared_ptr<CompactModelSourceGen> model_source_gen_{nullptr};

  std::ostringstream cache_;

  bool kernel_requires_id_argument_{true};

 public:
  // CompactCodeGen(CppAD::ADFun<CGBase> &fun, std::string model,
  //                bool function_only = false)
  //     : ModelSourceGen(fun, model), function_only_(function_only) {}

  CompactCodeGen(const std::string &model_name, bool function_only = false,
                 bool debug_mode = false)
      : model_name_(model_name),
        function_only_(function_only),
        debug_mode_(debug_mode) {}

  virtual ~CompactCodeGen() = default;

  virtual void set_tape(CppAD::ADFun<CGBase> &fun) {
    fun_ = &fun;
    model_source_gen_ =
        std::make_shared<CompactModelSourceGen>(fun, model_name_);
  }

  size_t input_dim() const {
    assert_model_source_gen();
    return fun_->Domain();
  }
  size_t &global_input_dim() { return global_input_dim_; }
  const size_t &global_input_dim() const { return global_input_dim_; }

  size_t local_input_dim() const {
    assert_model_source_gen();
    return fun_->Domain() - global_input_dim_;
  }
  size_t output_dim() const {
    assert_model_source_gen();
    return fun_->Range();
  }

  bool create_forward_zero() const;
  bool create_forward_zero(bool v);

  bool create_forward_one() const;
  bool create_forward_one(bool v);

  bool create_reverse_one() const;
  bool create_reverse_one(bool v);

  bool create_sparse_forward_one() const;
  bool create_sparse_forward_one(bool v);

  bool create_jacobian() const;
  bool create_jacobian(bool v);

  bool create_sparse_jacobian() const;
  bool create_sparse_jacobian(bool v);

  // returns the scalar type as string (e.g. "double")
  const std::string &base_type_name() const;

  bool is_function_only() const { return function_only_; }
  void set_function_only(bool option) { function_only_ = option; }

  bool debug_mode() const { return debug_mode_; }
  void set_debug_mode(bool d) { debug_mode_ = d; }

  AccumulationMethod &jacobian_acc_method() { return jac_acc_method_; }
  const AccumulationMethod &jacobian_acc_method() const {
    return jac_acc_method_;
  }

  const std::string &name() const { return model_name_; }

  /**
   * Name of the (primitive) type which is used for the scalars in the generated
   * code.
   */
  virtual std::string scalar_type() const { return "double"; }

  void set_jac_local_input_sparsity(const std::vector<size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < local_input_dim());
    }
    jac_local_input_sparsity_ = sparsity;
  }
  void set_jac_global_input_sparsity(const std::vector<size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < global_input_dim());
    }
    jac_global_input_sparsity_ = sparsity;
  }
  void set_jac_output_sparsity(const std::vector<size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < output_dim());
    }
    jac_output_sparsity_ = sparsity;
  }

  const std::map<std::string, std::string> &sources() const;

  virtual std::string sparse_jacobian_source();

  // std::string sparse_jacobian_source(
  //     const std::vector<size_t> &local_indices,
  //     const std::vector<size_t> &global_indices,
  //     AccumulationMethod acc_method = ACCUMULATE_MEAN) {
  //   const size_t output_dim = fun_->Range();
  //   std::vector<size_t> output_indices(output_dim, 0);
  //   std::iota(output_indices.begin(), output_indices.end(), 0);
  //   return sparse_jacobian_source(local_indices, global_indices,
  //   output_indices,
  //                                 acc_method);
  // }

  // std::string sparse_jacobian_source(
  //     const std::vector<size_t> &global_indices,
  //     AccumulationMethod acc_method = ACCUMULATE_MEAN) {
  //   const size_t input_dim = fun_->Domain() - global_input_dim_;
  //   std::vector<size_t> local_indices(input_dim, 0);
  //   std::iota(local_indices.begin(), local_indices.end(), 0);
  //   return sparse_jacobian_source(local_indices, global_indices, acc_method);
  // }

  virtual std::string jacobian_source();

  // std::string jacobian_source(const std::vector<size_t> &local_indices,
  //                             const std::vector<size_t> &global_indices,
  //                             AccumulationMethod acc_method =
  //                             ACCUMULATE_MEAN) {
  //   const size_t output_dim = fun_->Range();
  //   std::vector<size_t> output_indices(output_dim, 0);
  //   std::iota(output_indices.begin(), output_indices.end(), 0);
  //   return jacobian_source(local_indices, global_indices, output_indices,
  //                          acc_method);
  // }

  // std::string jacobian_source(const std::vector<size_t> &global_indices,
  //                             AccumulationMethod acc_method =
  //                             ACCUMULATE_MEAN) {
  //   const size_t input_dim = fun_->Domain() - global_input_dim_;
  //   std::vector<size_t> local_indices(input_dim, 0);
  //   std::iota(local_indices.begin(), local_indices.end(), 0);
  //   return jacobian_source(local_indices, global_indices, acc_method);
  // }

  /**
   * Generate library code for the forward zero pass.
   */
  virtual std::string forward_zero_source();

  /**
   * Generate library code for the forward one pass.
   */
  virtual std::string forward_one_source(
      std::vector<std::pair<std::string, std::string>> &sources);

  /**
   * Generate library code for the reverse one pass.
   */
  virtual std::string reverse_one_source(
      std::vector<std::pair<std::string, std::string>> &sources);

 protected:
  virtual std::string directional_forward_function_source(
      const std::string &function,
      const std::map<size_t, std::vector<size_t>> &elements,
      size_t input_dim) const;

  virtual std::string directional_reverse_function_source(
      const std::string &function,
      const std::map<size_t, std::vector<size_t>> &elements,
      size_t input_dim) const;

  virtual void generateSparseForwardOneSourcesWithAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);
  virtual void generateSparseForwardOneSourcesNoAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);

  virtual void generateSparseReverseOneSourcesWithAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);
  virtual void generateSparseReverseOneSourcesNoAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);

 public:
  /**
   * Emits the code for determining which thread the kernel execution is in (can
   * be left out if the thread ID is provided as argument to the kernel
   * function).
   */
  virtual void emit_thread_id_getter(std::ostringstream &code) const {}

  /**
   * Whether the kernel function should take the thread ID as an argument,
   * or whether it can be retrieved in some other way.
   */
  virtual bool kernel_requires_id_argument() const { return true; }

  /**
   * String to be prepended to the function signature.
   */
  virtual std::string function_type_prefix(bool is_function) const {
    return "";
  }

  virtual std::string header_file_extension() const { return ".h"; }
  virtual std::string source_file_extension() const { return ".cpp"; }

  virtual void emit_allocation_functions(std::ostringstream &code,
                                         const std::string &function_name,
                                         size_t local_input_dim,
                                         size_t global_input_dim,
                                         size_t output_dim) const = 0;
  virtual void emit_send_functions(std::ostringstream &code,
                                   const std::string &function_name,
                                   size_t local_input_dim,
                                   size_t global_input_dim,
                                   size_t output_dim) const = 0;
  virtual void emit_kernel_launch(std::ostringstream &code,
                                  const std::string &function_name,
                                  size_t local_input_dim,
                                  size_t global_input_dim,
                                  size_t output_dim) const = 0;

  virtual void emit_header(std::ostringstream &code,
                           const std::string &function_name,
                           size_t local_input_dim, size_t global_input_dim,
                           size_t output_dim,
                           AccumulationMethod acc_method) const;

  virtual void emit_function(std::ostringstream &code,
                             const std::string &function_name,
                             size_t local_input_dim, size_t global_input_dim,
                             size_t output_dim, const std::ostringstream &body,
                             LanguageCompact &language,
                             bool is_function = false,
                             bool is_forward_one = false,
                             bool is_reverse_one = false) const;

  virtual void emit_debug_print(std::ostringstream &code,
                                const std::string &function_name,
                                const std::string &var_vec_name,
                                const std::string &var_size_name,
                                const std::string &format = "%u") const;

  virtual void emit_cmake_code(std::ostringstream &code,
                               const std::string &project_name) const;

  virtual void emit_cpp_function_call_block(std::ostringstream &code,
                                            const std::string &fun_name,
                                            int output_dim,
                                            bool has_global_input) const;

  virtual void emit_cpp_function_call(std::ostringstream &code,
                                      const std::string &fun_name) const;

  virtual void emit_function_signature(std::ostringstream &code) const;

  // replaces repeating constants by variables to reduce the number of
  // characters
  static size_t replace_constants(std::string &code);

  virtual void assert_model_source_gen() const;
};

}  // namespace autogen