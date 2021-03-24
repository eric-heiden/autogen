namespace autogen {

template <class Base>
void CudaModelSourceGen<Base>::generateSparseForwardOneSourcesWithAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
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

    CppAD::cg::CodeHandler<Base> handler;
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

    std::ostringstream fun_body;

    LanguageCuda<Base> langC(false);
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_for1_indep" << j;
    // langC.setGenerateFunction(this->_cache.str());
    langC.setGenerateFunction("");

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dyCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<Base>> nameGen(
        this->createVariableNameGenerator("dy"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
        nameGen.get(), "dx", n);
    handler.generateCode(fun_body, langC, dyCustom, nameGenHess,
                         this->_atomicFunctions, subJobName);

    std::string fun_name = std::string(this->_name) +
                           "_sparse_forward_one_indep" + std::to_string(j);
    CudaFunctionSourceGen generator(fun_name, local_input_dim(),
                                    global_input_dim_, output_dim(),
                                    CUDA_ACCUMULATE_NONE);
    generator.is_forward_one = true;

    std::ostringstream complete;
    // complete << "__device__\n";
    // complete << fun_body.str();

    if (!kernel_only_) {
      generator.emit_header(complete);
    }
    //   LanguageCuda<Base> langC;
    generator.emit_kernel(complete, fun_body, langC, kernel_only_);
    if (!kernel_only_) {
      generator.emit_allocation_functions(complete);
      generator.emit_send_functions(complete);
      generator.emit_kernel_launch(complete);
    }

    std::string filename = fun_name + ".cuh";
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

template <class Base>
void CudaModelSourceGen<Base>::generateSparseForwardOneSourcesNoAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
  using std::vector;

  /**
   * Jacobian
   */
  size_t n = this->_fun.Domain();

  CppAD::cg::CodeHandler<Base> handler;
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
  std::map<size_t, vector<CGBase>> jac;                  // by column
  std::map<size_t, std::map<size_t, size_t>> positions;  // by column

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
  typename std::map<size_t, vector<CGBase>>::iterator itJ;
  for (itJ = jac.begin(); itJ != jac.end(); ++itJ) {
    size_t j = itJ->first;
    vector<CGBase>& dyCustom = itJ->second;

    this->_cache.str("");
    this->_cache << "model (forward one, indep " << j << ")";
    const std::string subJobName = this->_cache.str();

    std::ostringstream fun_body;

    LanguageCuda<Base> langC;
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_for1_indep" << j;
    langC.setGenerateFunction("");  // this->_cache.str());

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dyCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<Base>> nameGen(
        this->createVariableNameGenerator("dy"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
        nameGen.get(), "dx", n);
    handler.generateCode(fun_body, langC, dyCustom, nameGenHess,
                         this->_atomicFunctions, subJobName);

    // std::cout << code.str() << std::endl;

    std::string fun_name = std::string(this->_name) +
                           "_sparse_forward_one_indep" + std::to_string(j);
    CudaFunctionSourceGen generator(fun_name, local_input_dim(),
                                    global_input_dim_, output_dim(),
                                    CUDA_ACCUMULATE_NONE);
    generator.is_forward_one = true;

    std::ostringstream complete;

    if (!kernel_only_) {
      generator.emit_header(complete);
    }
    //   LanguageCuda<Base> langC;
    generator.emit_kernel(complete, fun_body, langC, kernel_only_);
    if (!kernel_only_) {
      generator.emit_allocation_functions(complete);
      generator.emit_send_functions(complete);
      generator.emit_kernel_launch(complete);
    }

    std::string filename = fun_name + ".cuh";
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

std::string directional_function_source(
    const std::string& function,
    const std::map<size_t, std::vector<size_t>>& elements, size_t input_dim) {
  std::stringstream code;
  std::string fun_title = "int " + function + "_sparse_forward_one(";
  code << "__device__\n";
  code << fun_title << "unsigned long pos,\n";
  code << std::string(fun_title.size(), ' ') << "Float* out,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* x,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* dx) {\n";
  // code << "  Float compressed[" << input_dim << "];\n";
  // code << "  unsigned long const* idx;\n";
  // code << "  unsigned long nnz;\n\n";

  code << "  switch(pos) {\n";
  for (const auto& it : elements) {
    // the size of each sparsity row
    code << "    case " << it.first
         << ":\n"
            "      "
         << function << "_sparse_forward_one_indep"
         << it.first
         //<< "(compressed, x, dx);\n"
         << "(out, x, dx);\n"
            "      return 0;\n";
  }
  code
      << "    default:\n"
         "      // printf(\"Error: cannot compute functional derivative %u for "
         "\\\""
      << function << "_sparse_forward_one\\\".\\n\", pos);\n";
  code << "      return 1;\n"
          "  };\n";

  // code << "  " << function << "_forward_one_sparsity(pos, &idx, &nnz);\n\n";

  // code << "  for (unsigned long ePos = 0; ePos < nnz; ePos++) {\n";
  // code << "    out[idx[ePos]] += compressed[ePos];\n";
  // code << "  }\n";

  code << "}\n";
  return code.str();
}

/**
 * Generate CUDA library code for the forward one pass.
 */
template <class Base>
std::string CudaModelSourceGen<Base>::forward_one_source(
    std::vector<std::pair<std::string, std::string>>& sources) {
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
  std::map<size_t, std::vector<size_t>> elements;
  for (size_t e = 0; e < this->_jacSparsity.rows.size(); e++) {
    elements[this->_jacSparsity.cols[e]].push_back(this->_jacSparsity.rows[e]);
  }

  std::cout << this->_name << " uses atomics? " << std::boolalpha
            << this->isAtomicsUsed() << std::endl;

  std::ostringstream code;

  if (this->isAtomicsUsed()) {
    generateSparseForwardOneSourcesWithAtomics(elements, code, sources);
  } else {
    generateSparseForwardOneSourcesNoAtomics(elements, code, sources);
  }

  code << "\n";

  const std::string sparsity_function =
      std::string(this->_name) + "_forward_one_sparsity";
  this->_cache.str("");
  this->generateSparsity1DSource2(sparsity_function, elements);
  code << "\n__device__\n" << this->_cache.str() << "\n";

  code << directional_function_source(this->_name, elements,
                                      this->_fun.Domain());

  size_t m = this->_fun.Range();
  size_t n = this->_fun.Domain();

  std::string sparse_for1_function =
      std::string(this->_name) + "_sparse_forward_one";
  const std::string model_function = std::string(this->_name) + "_forward_one";

  // logic from functor_generic_model.hpp:466 (ForwardOne)

  code << "\n__device__\n";
  LanguageCuda<Base>::printFunctionDeclaration(
      code, "int", model_function,
      {"Float *out", "const Float *x", "const Float *dx",
       "unsigned long nnzTx", "const unsigned long *idx"});
  code << " {\n"
       << "  unsigned long ePos, ej, j, nnz;\n"
       << "  int ret;\n"
       << "  unsigned long const* pos;\n"
       << "  Float compressed[" << n << "];\n"  // TODO should be m?
       << "  const Float *dx_in;\n\n";

  if (LanguageCuda<Base>::add_debug_prints) {
    code << "  printf(\"" << model_function
         << " idx:  \"); for (ej = 0; ej < nnzTx; ej++) printf(\"%u  \", "
            "idx[ej]); printf(\"\\n\");\n";
  }

#if 1
  //<< "  for (ePos = 0; ePos < " << n << "; ePos++) {\n"
  //<< "    out[ePos] = 0;\n"
  //<< "  }\n"

  code << "  for (ej = 0; ej < nnzTx; ej++) {\n"
       << "    j = idx[ej];\n"
       << "    " << sparsity_function << "(j, &pos, &nnz);\n"
       << "    if (nnz == 0) continue;\n"
       << "    dx_in = &dx[ej];\n"
       << "    ret = " << sparse_for1_function
       << "(j, compressed, x, dx_in);\n\n"
       << "    if (ret != 0) return ret;\n"
       << "    for (ePos = 0; ePos < nnz; ePos++) {\n"
       << "      if (ej == 0) out[pos[ePos]] = 0;\n"
       << "      out[pos[ePos]] += compressed[ePos];\n"
       << "    }\n"
       << "  }\n"
       << "  return 0;\n"
       << "}\n";
#else
  << "  unsigned long nnzMax = 0;\n"
  << "  unsigned long nnzTx = 0;\n"
  << "  unsigned long txPos[" << n << "];\n"
  << "  for (j = 0; j < " << n
  << "; j++) {\n"
  //<< "    ej = idx[j];\n"
  << "    if (dx[j] != 0.0) {\n"
  << "      " << sparsity_function << "(j, &pos, &nnz);\n"
  << "      if (nnz > nnzMax) nnzMax = nnz;\n"
  << "      else if (nnz == 0) continue;\n"
  << "      nnzTx++;\n"
  << "      txPos[nnzTx - 1] = j;\n"
  << "    }\n"
  << "  }\n\n"

  << "  for (j = 0; j < " << n << "; j++) {\n"
  << "    out[j] = 0;\n"
  << "  }\n"

  << "  for (ej = 0; ej < nnzTx; ej++) {\n"
  << "    j = txPos[ej];\n"
  << "    " << sparsity_function << "(j, &pos, &nnz);\n"
  << "    dx_in = &dx[j];\n"
  << "    " << sparse_for1_function << "(j, compressed, x, dx_in);\n\n"
  << "    for (ePos = 0; ePos < nnz; ePos++) {\n"
  << "      out[pos[ePos]] += compressed[ePos];\n"
  << "    }\n"
  << "  }\n"

  << "}\n";
#endif

#if 0  
  size_t m = this->_fun.Range();
  size_t n = this->_fun.Domain();

  const std::string model_function = std::string(this->_name) + "_forward_one";
  code << "__device__\n";
  LanguageCuda<Base>::printFunctionDeclaration(
      code, "int", model_function, {"Float ty[]", "Float const tx[]"});
  code << " {\n"
          "  unsigned long ePos, ej, i, j, nnz, nnzMax;\n"
          "  unsigned long const* pos;\n"
          "  unsigned long txPos["
       << n
       << "];\n"
          //   "  unsigned long* txPosTmp;\n"
          "  unsigned long nnzTx;\n"
          "  "
       << "Float const * in[2];\n"
          "  "
       << "Float* out[1];\n"
          "  "
       << "Float  x[" << n
       << "];\n"
          "  "
       << "Float compressed[" << n
       << "];\n"
          "  int ret;\n"
          "\n";


  code << "  printf(\"\\n\\n"
       << model_function << "\\n\");\n"
       << "  printf(\"tx[]:  \"); for (i = 0; i < " << n
       << ";++i) printf(\"%f   \", tx[i]); printf(\"\\n\");\n";



          //   "  txPos = 0;\n"
  code <<  "  nnzTx = 0;\n"
          "  nnzMax = 0;\n"
          "  for (j = 0; j < "
       << n
       << "; j++) {\n"
          //"    printf(\"j: %u   \", j);\n"
          //"    if (tx[j * 2 + 1] != 0.0) {\n"
          "    if (tx[j] != 0.0) {\n"
          "        "
       << sparsity_function
       << "(j, &pos, &nnz);\n"
          "      if (nnz > nnzMax)\n"
          "        nnzMax = nnz;\n"
          "      else if (nnz == 0)\n"
          "        continue;\n"
          "      nnzTx++;\n"
          "      printf(\""
       << model_function << "    nnzTx:  %u\\n\", nnzTx);\n"
        //   "        txPosTmp = (unsigned long*) realloc(txPos, nnzTx * "
        //   "sizeof(unsigned long));\n"
        //   "        if (txPosTmp != NULL) {\n"
        //   "           txPos = txPosTmp;\n"
        //   "        } else {\n"
        //   "           free(txPos);\n"
        //   "           return -1; // failure to allocate memory\n"
        //   "        }\n"
          "      txPos[nnzTx - 1] = j;\n"
          "    }\n"
          "  }\n"
          "  for (i = 0; i < "
       << m
       << "; i++) {\n"
          //"    ty[i * 2 + 1] = 0;\n"
          "    ty[i] = 0;\n"
          "  }\n"
          "\n"
        //   "  if (nnzTx == 0) {\n"
        //   "     free(txPos);\n"
        //   "     return 0; // nothing to do\n"
        //   "  }\n"
        //   "\n"
    //       "  compressed = ("
    //    << "Float *) malloc(nnzMax * sizeof(Float));\n"
          "\n"
          "  for (j = 0; j < "
       << n
       << "; j++)\n"



          //"    x[j] = tx[j * 2];\n"


          "    x[j] = tx[j];\n"


          "\n"
          "  for (ej = 0; ej < nnzTx; ej++) {\n"
          "    j = txPos[ej];\n"
          "    "
       << sparsity_function
       << "(j, &pos, &nnz);\n"
          "\n"
          "    in[0] = x;\n"
          "    in[1] = &tx[j * 2 + 1];\n"
          "    out[0] = compressed;\n";
  if (!this->_loopTapes.empty()) {
    code << "    for(ePos = 0; ePos < nnz; ePos++)\n"
            "      compressed[ePos] = 0;\n"
            "\n";
  }
  code << "     ret = " << sparse_for1_function << "(j, "
       << "out, in"  // args
       << ");\n"
          "\n"
          "    if (ret != 0) {\n"
        //   "        free(compressed);\n"
        //   "        free(txPos);\n"
          "      return ret;\n"
          "    }\n"
          "\n"
          "    for (ePos = 0; ePos < nnz; ePos++) {\n"
          "      ty[pos[ePos]] += compressed[ePos];\n"
          //"      ty[pos[ePos] * 2 + 1] += compressed[ePos];\n"
          "    }\n"
          "\n"
          "  }\n"
        //   "  free(compressed);\n"
        //   "  free(txPos);\n"
          "  return 0;\n"
          "}\n";
#endif

  return code.str();
}

}  // namespace autogen