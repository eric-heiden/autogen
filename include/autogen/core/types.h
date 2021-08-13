#pragma once

#include <cppad/cg.hpp>

namespace autogen {
using ADScalar = typename CppAD::AD<double>;
using CGScalar = typename CppAD::cg::CG<double>;
using ADCGScalar = typename CppAD::AD<CGScalar>;

using ADVector = std::vector<ADScalar>;
using ADCGVector = std::vector<ADCGScalar>;

using ADFun = typename CppAD::ADFun<double>;
using ADCGFun = typename CppAD::ADFun<CGScalar>;
}  // namespace autogen
