#include "autogen/cg/compact/compact_codegen.h"

namespace autogen {
/**
 * Helper class to expose some internal variables from
 * CppAD::cg::ModelCSourceGen.
 */
struct CompactModelSourceGen : public CppAD::cg::ModelCSourceGen<BaseScalar> {
  friend CompactCodeGen;

  using CGBase = typename CppAD::cg::CG<BaseScalar>;

  CompactModelSourceGen(CppAD::ADFun<CGBase> &fun, std::string model)
      : CppAD::cg::ModelCSourceGen<BaseScalar>(fun, model) {}

  //  expose some protected members

  CppAD::cg::JobTimer *getJobTimer() const { return this->_jobTimer; }

  const std::vector<BaseScalar> &getTypicalIndependentValues() const {
    return this->_x;
  }

  /**
   * The order of the atomic functions
   */
  const std::vector<std::string> &getAtomicFunctions() const {
    return this->_atomicFunctions;
  }
  std::vector<std::string> &getAtomicFunctions() {
    return this->_atomicFunctions;
  }

  const std::string &getBaseTypeName() const { return this->_baseTypeName; }

  /**
   * loop models
   */
  const std::set<CppAD::cg::LoopModel<BaseScalar> *> &getLoopTapes() const {
    return this->_loopTapes;
  }

  void setZeroEvaluated(bool zeroEvaluated) {
    this->_zeroEvaluated = zeroEvaluated;
  }
};
}  // namespace autogen