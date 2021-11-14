#pragma once

#include "types.hpp"

/**
 * Forward declarations of CppADCodeGen types that are needed for the back-end
 * to generate and compile models and libraries.
 */

namespace CppAD::cg {
template <class Base>
class ModelCSourceGen;

template <class Base>
class AbstractCCompiler;

template <class Base>
class MsvcCompiler;

template <class Base>
class ClangCompiler;

template <class Base>
class GccCompiler;

template <class Base>
class CGAtomicFunBridge;

template <class Base>
class WindowsDynamicLib;

template <class Base>
class LinuxDynamicLib;

template <class Base>
class LanguageC;

template <class Base>
class LanguageGenerationData;

template <class Base>
class VariableNameGenerator;

template <class Base>
class Argument;

template <class Base>
class OperationNode;

class IndexPattern;
class LinearIndexPattern;
class SectionedIndexPattern;
}  // namespace CppAD::cg

namespace autogen {
CppAD::cg::CGAtomicFunBridge<BaseScalar>* create_atomic_fun_bridge(
    const std::string& name, CppAD::ADFun<BaseScalar>& fun,
    bool standAlone = false, bool cacheSparsities = true);

CppAD::cg::CGAtomicFunBridge<BaseScalar>* create_atomic_fun_bridge(
    const std::string& name, CppAD::ADFun<CppAD::cg::CG<BaseScalar>>& fun,
    bool standAlone = false, bool cacheSparsities = true);

void call_atomic_fun_bridge(CppAD::cg::CGAtomicFunBridge<BaseScalar>* bridge,
                            const std::vector<ADCGScalar>& input,
                            std::vector<ADCGScalar>& output);
}  // namespace autogen