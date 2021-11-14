#pragma once

#include "base.hpp"
#include "cppadcg_user.hpp"

namespace autogen {
using ADScalar = typename CppAD::AD<BaseScalar>;
using CGScalar = typename CppAD::cg::CG<BaseScalar>;
using ADCGScalar = typename CppAD::AD<CGScalar>;

using ADVector = std::vector<ADScalar>;
using ADCGVector = std::vector<ADCGScalar>;

using ADFun = typename CppAD::ADFun<BaseScalar>;
using ADCGFun = typename CppAD::ADFun<CGScalar>;

enum ScalarType { SCALAR_DOUBLE, SCALAR_CPPAD, SCALAR_CODEGEN };

template <typename Scalar>
struct is_cppad_scalar {
  static constexpr bool value = false;
};
template <typename Scalar>
struct is_cppad_scalar<CppAD::AD<Scalar>> {
  static constexpr bool value = true;
};
template <typename Scalar>
struct is_cppad_scalar<CppAD::cg::CG<Scalar>> {
  static constexpr bool value = true;
};

template <typename Scalar>
struct is_cg_scalar {
  static constexpr bool value = false;
};
template <typename Scalar>
struct is_cg_scalar<CppAD::cg::CG<Scalar>> {
  static constexpr bool value = true;
};
template <>
struct is_cg_scalar<ADCGScalar> {
  static constexpr bool value = true;
};

static inline BaseScalar to_double(BaseScalar x) { return x; }
static inline BaseScalar to_double(const ADScalar& x) {
  return CppAD::Value(CppAD::Var2Par(x));
}
static inline BaseScalar to_double(const ADCGScalar& x) {
  return CppAD::Value(CppAD::Var2Par(x)).getValue();
}
}  // namespace autogen

#include "autogen/utils/conditionals.hpp"
