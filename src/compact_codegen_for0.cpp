#pragma once

#include "autogen/cg/compact/compact_codegen.h"

namespace autogen {

/**
 * Generate CUDA library code for the forward one pass.
 */
std::string CompactCodeGen::forward_zero_source() {
  const std::string jobName = "model (zero-order forward)";

  model_source_gen_->startingJob("'" + jobName + "'",
                                 CppAD::cg::JobTimer::GRAPH);

  CppAD::cg::CodeHandler<BaseScalar> handler;
  handler.setJobTimer(model_source_gen_->_jobTimer);

  if (global_input_dim() > fun_->Domain()) {
    throw std::runtime_error(
        "CUDA codegen failed: global data input size must not be "
        "larger than the provided input vector size.");
  }

  std::cout << "Generating code for function \"" << model_name_
            << "\" with input dimension " << input_dim()
            << " and output dimension " << output_dim() << "...\n";

  std::vector<CGBase> indVars(input_dim());
  handler.makeVariables(indVars);
  const auto& x = model_source_gen_->getTypicalIndependentValues();
  if (x.size() > 0) {
    for (std::size_t i = 0; i < indVars.size(); i++) {
      indVars[i].setValue(x[i]);
    }
  }

  std::vector<CGBase> dep;

  if (model_source_gen_->getLoopTapes().empty()) {
    dep = fun_->Forward(0, indVars);
  } else {
    /**
     * Contains loops
     */
    dep = model_source_gen_->prepareForward0WithLoops(handler, indVars);
  }
  
  model_source_gen_->setZeroEvaluated(true);
  model_source_gen_->finishedJob();

  LanguageCompact langC;
  langC.setMaxAssignmentsPerFunction(model_source_gen_->_maxAssignPerFunc,
                                     &model_source_gen_->_sources);
  langC.setMaxOperationsPerAssignment(
      model_source_gen_->_maxOperationsPerAssignment);
  langC.setParameterPrecision(model_source_gen_->_parameterPrecision);
  // set function name to empty string so that only the body gets generated
  langC.setGenerateFunction("");
  // langC.setGenerateFunction(model_name_ + "_forward_zero");

  std::ostringstream fun_body;
  CompactVariableNameGenerator nameGen(global_input_dim());

  handler.generateCode(fun_body, langC, dep, nameGen,
                       model_source_gen_->getAtomicFunctions(), jobName);
  langC.print_constants(fun_body);

  std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                              nameGen.getMinTemporaryVariableID();
  if (temporary_dim == 0) {
    std::cerr << "Warning: generated code for forward-zero pass of \""
              << model_name_ << "\" has no temporary variables.\n";
  } else {
    std::cout << "Code generated for forward-zero pass of \"" << model_name_
              << "\" with " << temporary_dim << " temporary variables.\n";
  }
  // for (const auto& var : nameGen.getTemporary()) {
  //   std::cout << "\t" << var.name << std::endl;
  // }

  std::string function_name = std::string(model_name_) + "_forward_zero";
  size_t out_dim = output_dim();

  std::ostringstream complete;
  complete << "#include \"util.h\"\n\n";

  if (!function_only_) {
    emit_header(complete, function_name, local_input_dim(), global_input_dim(),
                out_dim, ACCUMULATE_NONE);
  }
  //   LanguageCompact langC;
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