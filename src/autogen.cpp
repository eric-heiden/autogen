// clang-format off
#undef CPPAD_CPPAD_HPP
#include <cppad/cg.hpp>
#include <map>
#ifdef USE_EIGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#endif
#define CPPAD_CPPAD_HPP

#include "cppadcg_system.cpp"
#include "system.cpp"
#include "cache.cpp"
#include "compact_target.cpp"
#include "compact_codegen.cpp"

#include "nvcc_compiler.cpp"

#include "cuda_target.cpp"
#include "legacy_c_target.cpp"
#include "openmp_target.cpp"

#include "generated_codegen.cpp"
// clang-format on
