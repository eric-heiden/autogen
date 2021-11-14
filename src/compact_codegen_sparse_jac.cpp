#pragma once

#include "autogen/cg/compact/compact_codegen.h"

namespace autogen {
std::string CompactCodeGen::sparse_jacobian_source() {
  const std::string jobName = "sparse Jacobian";

  if (jac_local_input_sparsity_.empty() && jac_global_input_sparsity_.empty()) {
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

  std::vector<size_t> rows, cols;
  for (size_t output_i : jac_output_sparsity_) {
    for (size_t input_i : jac_global_input_sparsity_) {
      rows.push_back(output_i);
      cols.push_back(input_i);
    }
    for (size_t input_i : jac_local_input_sparsity_) {
      rows.push_back(output_i);
      cols.push_back(input_i + global_input_dim_);
    }
  }
  model_source_gen_->setCustomSparseJacobianElements(rows, cols);
  model_source_gen_->determineJacobianSparsity();

  // size_t m = _fun.Range();
  size_t n = fun_->Domain();

  model_source_gen_->startingJob("'" + jobName + "'",
                                 CppAD::cg::JobTimer::GRAPH);

  CppAD::cg::CodeHandler<BaseScalar> handler;
  handler.setJobTimer(model_source_gen_->getJobTimer());

  std::vector<CGBase> indVars(n);
  handler.makeVariables(indVars);
  const auto& x = model_source_gen_->getTypicalIndependentValues();
  if (x.size() > 0) {
    for (size_t i = 0; i < n; i++) {
      indVars[i].setValue(x[i]);
    }
  }

  const auto& sparsity = model_source_gen_->getJacobianSparsity();

  std::vector<CGBase> jac(sparsity.rows.size());
  bool forward = local_input_dim() + global_input_dim() <= output_dim();
  if (model_source_gen_->getLoopTapes().empty()) {
    // printSparsityPattern(sparsity.sparsity, "jac sparsity");
    CppAD::sparse_jacobian_work work;

    if (forward) {
      fun_->SparseJacobianForward(indVars, sparsity.sparsity, sparsity.rows,
                                  sparsity.cols, jac, work);
    } else {
      fun_->SparseJacobianReverse(indVars, sparsity.sparsity, sparsity.rows,
                                  sparsity.cols, jac, work);
    }
  } else {
    jac = model_source_gen_->prepareSparseJacobianWithLoops(handler, indVars,
                                                            forward);
  }

  model_source_gen_->finishedJob();

  LanguageCompact langC;
  langC.setMaxAssignmentsPerFunction(model_source_gen_->_maxAssignPerFunc,
                                     &model_source_gen_->_sources);
  langC.setMaxOperationsPerAssignment(
      model_source_gen_->_maxOperationsPerAssignment);
  langC.setParameterPrecision(model_source_gen_->_parameterPrecision);
  langC.setGenerateFunction("");  // _name + "_" + FUNCTION_SPARSE_JACOBIAN

  std::ostringstream fun_body;
  fun_body << "#include \"util.h\"\n\n";

  CompactVariableNameGenerator nameGen(global_input_dim_);

  // size_t arraySize = nameGen.getMaxTemporaryArrayVariableID();
  // size_t sArraySize = nameGen.getMaxTemporarySparseArrayVariableID();
  // if (arraySize > 0 || sArraySize > 0) {
  //   fun_body << "  Float* " << langC.auxArrayName_ << ";\n";
  // }

  // if (arraySize > 0 || sArraySize > 0 || zeroDependentArray) {
  //   _ss << _spaces << U_INDEX_TYPE << " i;\n";
  // }

  handler.generateCode(fun_body, langC, jac, nameGen,
                       model_source_gen_->getAtomicFunctions(), jobName);
  langC.print_constants(fun_body);

  size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                         nameGen.getMinTemporaryVariableID();
  if (temporary_dim == 0) {
    std::cerr << "Warning: the generated code has no temporary variables.\n";
  } else {
    std::cout << "Info: the generated code has " << temporary_dim
              << " temporary variables.\n";
  }

  std::ostringstream complete;
  complete << "#include \"util.h\"\n\n";

  size_t out_dim = rows.size();

  std::string function_name = model_name_ + "_sparse_jacobian";

  if (!function_only_) {
    emit_header(complete, function_name, local_input_dim(), global_input_dim(),
                out_dim, jac_acc_method_);
  }
  emit_function(complete, function_name, local_input_dim(), global_input_dim(),
                out_dim, fun_body, langC, function_only_);
  if (!function_only_) {
    emit_allocation_functions(complete, function_name, local_input_dim(),
                              global_input_dim(), out_dim);
    emit_send_functions(complete, function_name, local_input_dim(),
                        global_input_dim(), out_dim);
    emit_kernel_launch(complete, function_name, local_input_dim(),
                       global_input_dim(), out_dim);
  }

  return complete.str();
}
}  // namespace autogen