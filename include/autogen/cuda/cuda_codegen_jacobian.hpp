namespace autogen {

/**
 * Generate CUDA library code for the forward one pass.
 */
template <class Base>
std::string CudaModelSourceGen<Base>::jacobian_source() {
  const std::string jobName = "Jacobian";

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

  std::vector<CGBase> jac(this->_fun.Range() * this->_fun.Domain());
  jac = this->_fun.Jacobian(indVars);

  this->finishedJob();

  LanguageCuda<Base> langC;
  langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc, &this->_sources);
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
      std::string(this->_name) + "_jacobian", local_input_dim(),
      global_input_dim_, static_cast<int>(jac.size()), jac_acc_method_);

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

}  // namespace autogen