#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "autogen/autogen.hpp"

namespace py = pybind11;

using ADScalar = typename CppAD::AD<double>;
using CGScalar = typename CppAD::cg::CG<double>;
using ADCGScalar = typename CppAD::AD<CGScalar>;
using ADCGScalarPtr = typename std::shared_ptr<ADCGScalar>;

using ADVector = std::vector<ADScalar>;
using ADCGVector = std::vector<ADCGScalar>;
using ADCGPtrVector = std::vector<ADCGScalarPtr>;

using ADFun = typename CppAD::ADFun<double>;
using ADCGFun = typename CppAD::ADFun<CGScalar>;

PYBIND11_MAKE_OPAQUE(ADVector);
PYBIND11_MAKE_OPAQUE(ADCGVector);
PYBIND11_MAKE_OPAQUE(ADCGPtrVector);
PYBIND11_MAKE_OPAQUE(ADCGScalarPtr);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADFun>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADCGFun>);

template <typename Scalar>
py::class_<Scalar, std::shared_ptr<Scalar>> expose_scalar(py::handle m, const char* name) {
  return py::class_<Scalar, std::shared_ptr<Scalar>>(m, name)
      .def(py::init([](double t) { return Scalar(t); }))
      .def(py::init([](const Scalar& scalar) { return Scalar(scalar); }))
      .def(py::init())
      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self + float())
      .def(py::self - float())
      .def(py::self * float())
      .def(py::self / float())
      .def(float() + py::self)
      .def(float() - py::self)
      .def(float() * py::self)
      .def(float() / py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= py::self)
      .def(py::self /= py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__pow__",
           [](const Scalar& s, const Scalar& exponent) {
             return CppAD::pow(s, exponent);
           })
      .def("__pow__", [](const Scalar& s,
                         double exponent) { return CppAD::pow(s, exponent); })
      .def("abs", &Scalar::abs_me)
      .def("acos", &Scalar::acos_me)
      .def("asin", &Scalar::asin_me)
      .def("atan", &Scalar::atan_me)
      .def("cos", &Scalar::cos_me)
      .def("cosh", &Scalar::cosh_me)
      .def("exp", &Scalar::exp_me)
      .def("fabs", &Scalar::fabs_me)
      .def("log", &Scalar::log_me)
      .def("sin", &Scalar::sin_me)
      .def("sign", &Scalar::sign_me)
      .def("sinh", &Scalar::sinh_me)
      .def("sqrt", &Scalar::sqrt_me)
      .def("tan", &Scalar::tan_me)
      .def("tanh", &Scalar::tanh_me)
      .def("asinh", &Scalar::asinh_me)
      .def("acosh", &Scalar::acosh_me)
      .def("atanh", &Scalar::atanh_me)
      .def("erf", &Scalar::erf_me)
      .def("expm1", &Scalar::expm1_me)
      .def("log1p", &Scalar::log1p_me);
}

template <template <typename> typename Functor, typename Module>
void publish_function(Module m, const char* name) {
  static Functor<ADScalar> functor_ad;
  static Functor<ADCGScalar> functor_cg;
  // m.def(name,
  //       [&functor_ad](const ADScalar& x, const std::vector<ADScalar>& y)
  //           -> ADScalar { return functor_ad(x, y); });
//  m.def(name,
//        [&functor_cg](const ADCGScalar& x)
//            -> ADCGScalar {
//    std::cout << "Inside publish_function\n";
//    return functor_cg(x); });


//  static Functor<ADCGScalarPtr> functor_cg_ptr;
// Where does the conversion happen?
  m.def(name,
        [&functor_cg](const ADCGScalarPtr& x)
            -> ADCGScalarPtr {
    std::cout << "Inside pointer publish_function\n";
    return functor_cg(x); });
}