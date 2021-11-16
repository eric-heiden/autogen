#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <valarray>
#include <vector>

/**
 * The following includes are necessary for the user-facing CppAD + CppADCodeGen
 * functionalities.
 */

// clang-format off

// ---------------------------------------------------------------------------
// all base type requirements
#include <cppad/base_require.hpp>

// ---------------------------------------------------------------------------
#include <cppad/cg/cppadcg_assert.hpp>
#include <cppad/cg/exception.hpp>
#include <cppad/cg/operation.hpp>
#include <cppad/cg/declare_cg.hpp>

// ---------------------------------------------------------------------------
// some utilities
#include <cppad/cg/smart_containers.hpp>
#include <cppad/cg/ostream_config_restore.hpp>
#include <cppad/cg/array_view.hpp>

// ---------------------------------------------------------------------------
// indexes
#include <cppad/cg/patterns/index/index_pattern.hpp>
#include <cppad/cg/patterns/index/linear_index_pattern.hpp>
#include <cppad/cg/patterns/index/sectioned_index_pattern.hpp>
#include <cppad/cg/patterns/index/plane_2d_index_pattern.hpp>
#include <cppad/cg/patterns/index/random_index_pattern.hpp>
#include <cppad/cg/patterns/index/random_1d_index_pattern.hpp>
#include <cppad/cg/patterns/index/random_2d_index_pattern.hpp>
#include <cppad/cg/patterns/index/index_pattern_impl.hpp>

// ---------------------------------------------------------------------------
// core files
#include <cppad/cg/debug.hpp>
#include <cppad/cg/argument.hpp>
#include <cppad/cg/operation_node.hpp>
#include <cppad/cg/operation_stack.hpp>
#include <cppad/cg/nodes/index_operation_node.hpp>
#include <cppad/cg/nodes/index_assign_operation_node.hpp>
#include <cppad/cg/nodes/loop_start_operation_node.hpp>
#include <cppad/cg/nodes/loop_end_operation_node.hpp>
#include <cppad/cg/nodes/print_operation_node.hpp>
#include <cppad/cg/cg.hpp>
#include <cppad/cg/default.hpp>
#include <cppad/cg/variable.hpp>
#include <cppad/cg/identical.hpp>
#include <cppad/cg/range.hpp>
#include <cppad/cg/atomic_dependency_locator.hpp>
#include <cppad/cg/variable_name_generator.hpp>
#include <cppad/cg/job_timer.hpp>
#include <cppad/cg/lang/language.hpp>
#include <cppad/cg/lang/lang_stream_stack.hpp>
#include <cppad/cg/scope_path_element.hpp>
#include <cppad/cg/array_id_compresser.hpp>
#include <cppad/cg/patterns/loop_position.hpp>
#include <cppad/cg/arithmetic.hpp>
#include <cppad/cg/arithmetic_assign.hpp>
#include <cppad/cg/math.hpp>
#include <cppad/cg/math_other.hpp>
#include <cppad/cg/nan.hpp>
#include <cppad/cg/cond_exp_op.hpp>
#include <cppad/cg/compare.hpp>
#include <cppad/cg/ordered.hpp>
#include <cppad/cg/unary.hpp>

// ---------------------------------------------------------------------------
#include <cppad/cg/code_handler.hpp>
#include <cppad/cg/code_handler_impl.hpp>
#include <cppad/cg/code_handler_vector.hpp>
#include <cppad/cg/code_handler_loops.hpp>

// ---------------------------------------------------------------------------
#include <cppad/cg/base_double.hpp>
#include <cppad/cg/base_float.hpp>

// ---------------------------------------------------------------------------
// CppAD
#include <cppad/cppad.hpp>

// resolves some ambiguities
#include <cppad/cg/arithmetic_ad.hpp>

// addons
#include <cppad/cg/extra/extra.hpp>

// ---------------------------------------------------------------------------
// additional utilities
#include <cppad/cg/util.hpp>

// ---------------------------------------------------------------------------
// atomic function utilities
#include <cppad/cg/custom_position.hpp>
#include <cppad/cg/base_abstract_atomic_fun.hpp>
#include <cppad/cg/abstract_atomic_fun.hpp>

// clang-format on