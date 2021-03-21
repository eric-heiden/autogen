namespace autogen {

template <class Base>
void CudaModelSourceGen<Base>::generateSparseForwardOneSourcesWithAtomics(
    const std::map<size_t, std::vector<size_t> >& elements,
    std::ostringstream& code, CudaVariableNameGenerator<Base>& nameGen,
    LanguageCuda<Base>& langC, CppAD::cg::CodeHandler<Base> &handler) {
  using std::vector;

  /**
   * Generate one function for each dependent variable
   */
  size_t n = this->_fun.Domain();

  vector<CGBase> dxv(n);

  const std::string jobName = "model (forward one)";
  this->startingJob("'" + jobName + "'",
                    CppAD::cg::JobTimer::SOURCE_GENERATION);

  for (const auto& it : elements) {
    size_t j = it.first;
    const std::vector<size_t>& rows = it.second;

    this->_cache.str("");
    this->_cache << "model (forward one, indep " << j << ")";
    const std::string subJobName = this->_cache.str();

    this->startingJob("'" + subJobName + "'", CppAD::cg::JobTimer::GRAPH);

    // CppAD::cg::CodeHandler<Base> handler;
    handler.setJobTimer(this->_jobTimer);

    vector<CGBase> indVars(n);
    handler.makeVariables(indVars);
    if (this->_x.size() > 0) {
      for (size_t i = 0; i < n; i++) {
        indVars[i].setValue(this->_x[i]);
      }
    }

    CGBase dx;
    handler.makeVariable(dx);
    if (this->_x.size() > 0) {
      dx.setValue(Base(1.0));
    }

    // TODO: consider caching the zero order coefficients somehow between calls
    this->_fun.Forward(0, indVars);
    dxv[j] = dx;
    vector<CGBase> dy = this->_fun.Forward(1, dxv);
    dxv[j] = Base(0);
    CPPADCG_ASSERT_UNKNOWN(dy.size() == this->_fun.Range());

    vector<CGBase> dyCustom;
    for (size_t it2 : rows) {
      dyCustom.push_back(dy[it2]);
    }

    this->finishedJob();

    // LanguageCuda<Base> langC;
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_for1_indep" << j;
    // langC.setGenerateFunction(this->_cache.str());
    langC.setGenerateFunction("");

    // std::ostringstream code;

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    handler.generateCode(code, langC, dyCustom, nameGen, this->_atomicFunctions,
                         subJobName);

    // std::unique_ptr<CppAD::cg::VariableNameGenerator<Base> > nameGen(
    //     this->createVariableNameGenerator("dy"));
    // CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
    //     nameGen.get(), "dx", n);
    // handler.generateCode(code, langC, dyCustom, nameGenHess,
    //                      this->_atomicFunctions, subJobName);

    std::cout << code.str() << std::endl;
  }
}

template <class Base>
void CudaModelSourceGen<Base>::generateSparseForwardOneSourcesNoAtomics(
    const std::map<size_t, std::vector<size_t> >& elements,
    std::ostringstream& code, CudaVariableNameGenerator<Base>& nameGen,
    LanguageCuda<Base>& langC, CppAD::cg::CodeHandler<Base> &handler) {
  using std::vector;

  /**
   * Jacobian
   */
  size_t n = this->_fun.Domain();

//   CppAD::cg::CodeHandler<Base> handler;
  handler.setJobTimer(this->_jobTimer);

  vector<CGBase> x(n);
  handler.makeVariables(x);
  if (this->_x.size() > 0) {
    for (size_t i = 0; i < n; i++) {
      x[i].setValue(this->_x[i]);
    }
  }

  CGBase dx;
  handler.makeVariable(dx);
  if (this->_x.size() > 0) {
    dx.setValue(Base(1.0));
  }

  vector<CGBase> jacFlat(this->_jacSparsity.rows.size());

  CppAD::sparse_jacobian_work work;  // temporary structure for CPPAD
  this->_fun.SparseJacobianForward(x, this->_jacSparsity.sparsity,
                                   this->_jacSparsity.rows,
                                   this->_jacSparsity.cols, jacFlat, work);

  /**
   * organize results
   */
  std::map<size_t, vector<CGBase> > jac;                  // by column
  std::map<size_t, std::map<size_t, size_t> > positions;  // by column

  for (const auto& it : elements) {
    size_t j = it.first;
    const std::vector<size_t>& column = it.second;

    jac[j].resize(column.size());
    std::map<size_t, size_t>& pos = positions[j];

    for (size_t e = 0; e < column.size(); e++) {
      size_t i = column[e];
      pos[i] = e;
    }
  }

  for (size_t el = 0; el < this->_jacSparsity.rows.size(); el++) {
    size_t i = this->_jacSparsity.rows[el];
    size_t j = this->_jacSparsity.cols[el];
    size_t e = positions[j].at(i);

    vector<CGBase>& column = jac[j];
    column[e] = jacFlat[el] * dx;
  }

  /**
   * Create source for each independent/column
   */
  typename std::map<size_t, vector<CGBase> >::iterator itJ;
  for (itJ = jac.begin(); itJ != jac.end(); ++itJ) {
    size_t j = itJ->first;
    vector<CGBase>& dyCustom = itJ->second;

    this->_cache.str("");
    this->_cache << "model (forward one, indep " << j << ")";
    const std::string subJobName = this->_cache.str();

    // LanguageCuda<Base> langC;
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_for1_indep" << j;
    langC.setGenerateFunction("");  // this->_cache.str());

    // std::ostringstream code;

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    handler.generateCode(code, langC, dyCustom, nameGen, this->_atomicFunctions,
                         subJobName);

    // std::unique_ptr<CppAD::cg::VariableNameGenerator<Base> > nameGen(
    //     this->createVariableNameGenerator("dy"));
    // CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
    //     nameGen.get(), "dx", n);
    // handler.generateCode(code, langC, dyCustom, nameGenHess,
    //                      this->_atomicFunctions, subJobName);

    std::cout << code.str() << std::endl;
  }
}

/**
 * Generate CUDA library code for the forward one pass.
 */
template <class Base>
std::string CudaModelSourceGen<Base>::forward_one_source() {
  const std::string jobName = "model (first-order forward)";

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

  std::cout << "Generating first-order forward code for function \""
            << this->_name << "\" with input dimension " << local_input_dim
            << " and output dimension " << output_dim << "...\n";

//   std::vector<CGBase> indVars(local_input_dim + global_input_dim_);
//   handler.makeVariables(indVars);
//   if (this->_x.size() > 0) {
//     for (std::size_t i = 0; i < indVars.size(); i++) {
//       indVars[i].setValue(this->_x[i]);
//     }
//   }

//   std::vector<CGBase> dep;

  this->determineJacobianSparsity();
  // elements[var]{equations}
  std::map<size_t, std::vector<size_t> > elements;
  for (size_t e = 0; e < this->_jacSparsity.rows.size(); e++) {
    elements[this->_jacSparsity.cols[e]].push_back(this->_jacSparsity.rows[e]);
  }

  std::cout << this->_name << " uses atomics? " << std::boolalpha
            << this->isAtomicsUsed() << std::endl;

  std::ostringstream code;

  CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
  LanguageCuda<Base> langC;
  if (this->isAtomicsUsed()) {
    generateSparseForwardOneSourcesWithAtomics(elements, code, nameGen, langC, handler);
  } else {
    generateSparseForwardOneSourcesNoAtomics(elements, code, nameGen, langC, handler);
  }

  std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                              nameGen.getMinTemporaryVariableID();
  if (temporary_dim == 0) {
    std::cerr << "Warning: generated code has no temporary variables.\n";
  } else {
    std::cout << "Code generated with " << temporary_dim
              << " temporary variables.\n";
  }
  // for (const auto& var : nameGen.getTemporary()) {
  //   std::cout << "\t" << var.name << std::endl;
  // }

  CudaFunctionSourceGen generator(std::string(this->_name) + "_forward_one",
                                  local_input_dim, global_input_dim_,
                                  output_dim, CUDA_ACCUMULATE_NONE);

  std::ostringstream complete;

  if (!kernel_only_) {
    generator.emit_header(complete);
  }
//   LanguageCuda<Base> langC;
  generator.emit_kernel(complete, code, langC, kernel_only_);
  if (!kernel_only_) {
    generator.emit_allocation_functions(complete);
    generator.emit_send_functions(complete);
    generator.emit_kernel_launch(complete);
  }

  return complete.str();
}

}  // namespace autogen