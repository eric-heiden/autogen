#pragma once

#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>
#include <numeric>

#include "cuda_function_sourcegen.hpp"

namespace autogen {
template <class Base>
class CudaModelSourceGen : public CppAD::cg::ModelCSourceGen<Base> {
  using CGBase = CppAD::cg::CG<Base>;

 protected:
  std::size_t global_input_dim_{0};

  AccumulationMethod jac_acc_method_{ACCUMULATE_MEAN};

  std::vector<std::size_t> jac_local_input_sparsity_;
  std::vector<std::size_t> jac_global_input_sparsity_;
  std::vector<std::size_t> jac_output_sparsity_;

  /**
   * Whether to only generate the CUDA kernel, not the kernel launch and memory
   * access functions.
   */
  bool kernel_only_{false};

 public:
  CudaModelSourceGen(CppAD::ADFun<CppAD::cg::CG<Base>> &fun, std::string model,
                     bool kernel_only = false)
      : CppAD::cg::ModelCSourceGen<Base>(fun, model),
        kernel_only_(kernel_only) {}

  std::size_t &global_input_dim() { return global_input_dim_; }
  const std::size_t &global_input_dim() const { return global_input_dim_; }

  std::size_t local_input_dim() const {
    return this->_fun.Domain() - global_input_dim_;
  }
  std::size_t output_dim() const { return this->_fun.Range(); }

  std::string base_type_name() const { return this->_baseTypeName; }

  bool is_kernel_only() const { return kernel_only_; }
  void set_kernel_only(bool option) { kernel_only_ = option; }

  AccumulationMethod &jacobian_acc_method() { return jac_acc_method_; }
  const AccumulationMethod &jacobian_acc_method() const {
    return jac_acc_method_;
  }

  void set_jac_local_input_sparsity(const std::vector<std::size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < local_input_dim());
    }
    jac_local_input_sparsity_ = sparsity;
  }
  void set_jac_global_input_sparsity(const std::vector<std::size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < global_input_dim());
    }
    jac_global_input_sparsity_ = sparsity;
  }
  void set_jac_output_sparsity(const std::vector<std::size_t> &sparsity) {
    for (auto idx : sparsity) {
      assert(idx < output_dim());
    }
    jac_output_sparsity_ = sparsity;
  }

  const std::map<std::string, std::string> &sources() {
    auto mtt = CppAD::cg::MultiThreadingType::NONE;
    CppAD::cg::JobTimer *timer = nullptr;
    return this->getSources(mtt, timer);
  }

  std::string sparse_jacobian_source() {
    const std::string jobName = "sparse Jacobian";

    if (jac_local_input_sparsity_.empty() &&
        jac_global_input_sparsity_.empty()) {
      // assume dense Jacobian
      jac_local_input_sparsity_.resize(local_input_dim());
      std::iota(jac_local_input_sparsity_.begin(),
                jac_local_input_sparsity_.end(), 0);
      jac_global_input_sparsity_.resize(global_input_dim());
      std::iota(jac_global_input_sparsity_.begin(),
                jac_global_input_sparsity_.end(), 0);
    }
    if (jac_output_sparsity_.empty()) {
      jac_output_sparsity_.resize(output_dim());
      std::iota(jac_output_sparsity_.begin(), jac_output_sparsity_.end(), 0);
    }

    std::vector<std::size_t> rows, cols;
    for (std::size_t output_i : jac_output_sparsity_) {
      for (std::size_t input_i : jac_global_input_sparsity_) {
        rows.push_back(output_i);
        cols.push_back(input_i);
      }
      for (std::size_t input_i : jac_local_input_sparsity_) {
        rows.push_back(output_i);
        cols.push_back(input_i + global_input_dim_);
      }
    }
    this->setCustomSparseJacobianElements(rows, cols);
    this->determineJacobianSparsity();

    // size_t m = _fun.Range();
    std::size_t n = this->_fun.Domain();

    this->startingJob("'" + jobName + "'", CppAD::cg::JobTimer::GRAPH);

    CppAD::cg::CodeHandler<Base> handler;
    handler.setJobTimer(this->_jobTimer);

    std::vector<CGBase> indVars(n);
    handler.makeVariables(indVars);
    if (this->_x.size() > 0) {
      for (size_t i = 0; i < n; i++) {
        indVars[i].setValue(this->_x[i]);
      }
    }

    std::vector<CGBase> jac(this->_jacSparsity.rows.size());
    bool forward = local_input_dim() + global_input_dim() <= output_dim();
    if (this->_loopTapes.empty()) {
      // printSparsityPattern(this->_jacSparsity.sparsity, "jac sparsity");
      CppAD::sparse_jacobian_work work;

      if (forward) {
        this->_fun.SparseJacobianForward(indVars, this->_jacSparsity.sparsity,
                                         this->_jacSparsity.rows,
                                         this->_jacSparsity.cols, jac, work);
      } else {
        this->_fun.SparseJacobianReverse(indVars, this->_jacSparsity.sparsity,
                                         this->_jacSparsity.rows,
                                         this->_jacSparsity.cols, jac, work);
      }
    } else {
      jac = this->prepareSparseJacobianWithLoops(handler, indVars, forward);
    }

    this->finishedJob();

    LanguageCuda<Base> langC;
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    langC.setGenerateFunction("");  // _name + "_" + FUNCTION_SPARSE_JACOBIAN

    std::ostringstream code;

    CudaVariableNameGenerator<Base> nameGen(global_input_dim_);

    // size_t arraySize = nameGen.getMaxTemporaryArrayVariableID();
    // size_t sArraySize = nameGen.getMaxTemporarySparseArrayVariableID();
    // if (arraySize > 0 || sArraySize > 0) {
    //   code << "  Float* " << langC.auxArrayName_ << ";\n";
    // }

    // if (arraySize > 0 || sArraySize > 0 || zeroDependentArray) {
    //   _ss << _spaces << U_INDEX_TYPE << " i;\n";
    // }

    handler.generateCode(code, langC, jac, nameGen, this->_atomicFunctions,
                         jobName);
    langC.print_constants(code);

    std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                                nameGen.getMinTemporaryVariableID();
    if (temporary_dim == 0) {
      std::cerr << "Warning: the generated code has no temporary variables.\n";
    } else {
      std::cout << "Info: the generated code has " << temporary_dim
                << " temporary variables.\n";
    }

    std::ostringstream complete;

    CudaFunctionSourceGen generator(
        std::string(this->_name) + "_sparse_jacobian", local_input_dim(),
        global_input_dim_, rows.size(), jac_acc_method_);

    if (!kernel_only_) {
      generator.emit_header(complete);
    }
    generator.emit_kernel(complete, code, langC, kernel_only_);
    if (!kernel_only_) {
      generator.emit_allocation_functions(complete);
      generator.emit_send_functions(complete);
      generator.emit_kernel_launch(complete);
    }

    return complete.str();
  }

  std::string sparse_jacobian_source(
      const std::vector<std::size_t> &local_indices,
      const std::vector<std::size_t> &global_indices,
      AccumulationMethod acc_method = ACCUMULATE_MEAN) {
    const std::size_t output_dim = this->_fun.Range();
    std::vector<size_t> output_indices(output_dim, 0);
    std::iota(output_indices.begin(), output_indices.end(), 0);
    return sparse_jacobian_source(local_indices, global_indices, output_indices,
                                  acc_method);
  }

  std::string sparse_jacobian_source(
      const std::vector<std::size_t> &global_indices,
      AccumulationMethod acc_method = ACCUMULATE_MEAN) {
    const std::size_t input_dim = this->_fun.Domain() - global_input_dim_;
    std::vector<size_t> local_indices(input_dim, 0);
    std::iota(local_indices.begin(), local_indices.end(), 0);
    return sparse_jacobian_source(local_indices, global_indices, acc_method);
  }

  std::string jacobian_source();

  std::string jacobian_source(
      const std::vector<std::size_t> &local_indices,
      const std::vector<std::size_t> &global_indices,
      AccumulationMethod acc_method = ACCUMULATE_MEAN) {
    const std::size_t output_dim = this->_fun.Range();
    std::vector<size_t> output_indices(output_dim, 0);
    std::iota(output_indices.begin(), output_indices.end(), 0);
    return jacobian_source(local_indices, global_indices, output_indices,
                           acc_method);
  }

  std::string jacobian_source(
      const std::vector<std::size_t> &global_indices,
      AccumulationMethod acc_method = ACCUMULATE_MEAN) {
    const std::size_t input_dim = this->_fun.Domain() - global_input_dim_;
    std::vector<size_t> local_indices(input_dim, 0);
    std::iota(local_indices.begin(), local_indices.end(), 0);
    return jacobian_source(local_indices, global_indices, acc_method);
  }

  /**
   * Generate CUDA library code for the forward zero pass.
   */
  std::string forward_zero_source();

  /**
   * Generate CUDA library code for the forward one pass.
   */
  std::string forward_one_source(
      std::vector<std::pair<std::string, std::string>> &sources);

  /**
   * Generate CUDA library code for the reverse one pass.
   */
  std::string reverse_one_source(
      std::vector<std::pair<std::string, std::string>> &sources);

 protected:
  void generateSparseForwardOneSourcesWithAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);
  void generateSparseForwardOneSourcesNoAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);

  void generateSparseReverseOneSourcesWithAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);
  void generateSparseReverseOneSourcesNoAtomics(
      const std::map<size_t, std::vector<size_t>> &elements,
      std::ostringstream &code,
      std::vector<std::pair<std::string, std::string>> &sources);
};

}  // namespace autogen

#include "cuda_codegen_for0.hpp"
#include "cuda_codegen_for1.hpp"
#include "cuda_codegen_jacobian.hpp"
#include "cuda_codegen_rev1.hpp"