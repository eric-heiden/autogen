namespace autogen {

/**
 * Generate CUDA library code for the forward one pass.
 */
template <class Base>
std::string CudaModelSourceGen<Base>::forward_zero_source() {
  const std::string jobName = "model (zero-order forward)";

  this->startingJob("'" + jobName + "'", CppAD::cg::JobTimer::GRAPH);

  CppAD::cg::CodeHandler<Base> handler;
  handler.setJobTimer(this->_jobTimer);

  if (global_input_dim_ > this->_fun.Domain()) {
    throw std::runtime_error(
        "CUDA codegen failed: global data input size must not be "
        "larger than the provided input vector size.");
  }

  const std::size_t local_input_dim = this->_fun.Domain() - global_input_dim_;
  const std::size_t output_dim = this->_fun.Range();

  std::cout << "Generating code for function \"" << this->_name
            << "\" with input dimension " << local_input_dim
            << " and output dimension " << output_dim << "...\n";

  std::vector<CGBase> indVars(local_input_dim + global_input_dim_);
  handler.makeVariables(indVars);
  if (this->_x.size() > 0) {
    for (std::size_t i = 0; i < indVars.size(); i++) {
      indVars[i].setValue(this->_x[i]);
    }
  }

  std::vector<CGBase> dep;

  if (this->_loopTapes.empty()) {
    dep = this->_fun.Forward(0, indVars);
  } else {
    /**
     * Contains loops
     */
    dep = this->prepareForward0WithLoops(handler, indVars);
  }
  this->_zeroEvaluated = true;

  this->finishedJob();

  LanguageCuda<Base> langC;
  langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc, &this->_sources);
  langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
  langC.setParameterPrecision(this->_parameterPrecision);
  // set function name to empty string so that only the body gets generated
  langC.setGenerateFunction("");
  // langC.setGenerateFunction(this->_name + "_forward_zero");

  std::ostringstream code;
  CudaVariableNameGenerator<Base> nameGen(global_input_dim_);

  handler.generateCode(code, langC, dep, nameGen, this->_atomicFunctions,
                       jobName);
  langC.print_constants(code);

  std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                              nameGen.getMinTemporaryVariableID();
  if (temporary_dim == 0) {
    std::cerr << "Warning: generated code for forward-zero pass of \""
              << this->_name << "\" has no temporary variables.\n";
  } else {
    std::cout << "Code generated for forward-zero pass of \"" << this->_name
              << "\" with " << temporary_dim << " temporary variables.\n";
  }
  // for (const auto& var : nameGen.getTemporary()) {
  //   std::cout << "\t" << var.name << std::endl;
  // }

  CudaFunctionSourceGen generator(std::string(this->_name) + "_forward_zero",
                                  local_input_dim, global_input_dim_,
                                  output_dim, ACCUMULATE_NONE);

  std::ostringstream complete;

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