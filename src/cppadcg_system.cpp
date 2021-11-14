#include "autogen/core/cppadcg_system.h"

namespace autogen {
// CppAD::cg::CGAtomicFunBridge<BaseScalar>* create_atomic_fun_bridge(
//     const std::string& name, CppAD::ADFun<BaseScalar>& fun, bool standAlone,
//     bool cacheSparsities) {
//   return new CppAD::cg::CGAtomicFunBridge<BaseScalar>(name, fun, standAlone,
//                                                       cacheSparsities);
// }

CppAD::cg::CGAtomicFunBridge<BaseScalar>* create_atomic_fun_bridge(
    const std::string& name, CppAD::ADFun<CppAD::cg::CG<BaseScalar>>& fun,
    bool standAlone, bool cacheSparsities) {
  return new CppAD::cg::CGAtomicFunBridge<BaseScalar>(name, fun, standAlone,
                                                      cacheSparsities);
}

void call_atomic_fun_bridge(
    CppAD::cg::CGAtomicFunBridge<BaseScalar>* bridge,
    const std::vector<CppAD::AD<CppAD::cg::CG<BaseScalar>>>& input,
    std::vector<CppAD::AD<CppAD::cg::CG<BaseScalar>>>& output) {
  (*bridge)(input, output);
}
}  // namespace autogen