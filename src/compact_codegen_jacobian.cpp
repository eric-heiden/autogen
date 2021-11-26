#pragma once

#include "autogen/cg/compact/compact_codegen.h"

namespace autogen {

/**
 * Generate CUDA library code for the forward one pass.
 */
std::string CompactCodeGen::jacobian_source() {
  const std::string jobName = "Jacobian";

  // size_t m = _fun.Range();
  std::size_t n = fun_->Domain();

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

  std::vector<CGBase> jac(fun_->Range() * fun_->Domain());
  jac = fun_->Jacobian(indVars);

  model_source_gen_->finishedJob();

  LanguageCompact langC;
  langC.setMaxAssignmentsPerFunction(model_source_gen_->_maxAssignPerFunc,
                                     &model_source_gen_->_sources);
  langC.setMaxOperationsPerAssignment(
      model_source_gen_->_maxOperationsPerAssignment);
  langC.setParameterPrecision(model_source_gen_->_parameterPrecision);
  langC.setGenerateFunction("");  // _name + "_" + FUNCTION_SPARSE_JACOBIAN

  std::ostringstream code;

  CompactVariableNameGenerator nameGen(global_input_dim_);

  // size_t arraySize = nameGen.getMaxTemporaryArrayVariableID();
  // size_t sArraySize = nameGen.getMaxTemporarySparseArrayVariableID();
  // if (arraySize > 0 || sArraySize > 0) {
  //   code << "  Float* " << langC.auxArrayName_ << ";\n";
  // }

  // if (arraySize > 0 || sArraySize > 0 || zeroDependentArray) {
  //   _ss << _spaces << U_INDEX_TYPE << " i;\n";
  // }

  handler.generateCode(code, langC, jac, nameGen,
                       model_source_gen_->getAtomicFunctions(), jobName);

  langC.print_constants(code);

  std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                              nameGen.getMinTemporaryVariableID();
  if (temporary_dim == 0) {
    std::cerr << "Warning: generated code for Jacobian pass of \""
              << model_name_ << "\" has no temporary variables.\n";
  } else {
#ifdef DEBUG
    std::cout << "Code generated for Jacobian pass of \"" << model_name_
              << "\" with " << temporary_dim << " temporary variables.\n";
#endif
  }

  std::ostringstream complete;
  complete << "#include \"util.h\"\n\n";

  std::string function_name = std::string(model_name_) + "_jacobian";
  size_t out_dim = jac.size();

  if (!function_only_) {
    emit_header(complete, function_name, local_input_dim(), global_input_dim(),
                out_dim, jac_acc_method_);
  }
  emit_function(complete, function_name, local_input_dim(), global_input_dim(),
                out_dim, code, langC, function_only_);
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