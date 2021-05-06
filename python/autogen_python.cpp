#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "autogen/autogen_lightweight.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using CppADScalar = typename CppAD::AD<double>;
using CGScalar = typename CppAD::AD<CppAD::cg::CG<double>>;
using ADVector = std::vector<CppADScalar>;
using CGVector = std::vector<CGScalar>;

using ADFun = typename CppAD::ADFun<CppAD::cg::CG<double>>;

PYBIND11_MAKE_OPAQUE(ADVector);
PYBIND11_MAKE_OPAQUE(CGVector);

PYBIND11_MODULE(_autogen, m) {
  m.doc() = R"pbdoc(
        Autogen python plugin
        -----------------------

        .. currentmodule:: autogen

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  py::bind_vector<ADVector>(m, "CppADVector");
  py::bind_vector<CGVector>(m, "CGVector");

  py::class_<CppADScalar>(m, "CppADScalar")
      .def(py::init([](double t) { return CppADScalar(t); }))
      .def(py::init(
          [](const CppADScalar& scalar) { return CppADScalar(scalar); }))
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
           [](const CppADScalar& s, const CppADScalar& exponent) {
             return CppAD::pow(s, exponent);
           })
      .def("__pow__", [](const CppADScalar& s,
                         double exponent) { return CppAD::pow(s, exponent); })
      .def("abs", &CppADScalar::abs_me)
      .def("acos", &CppADScalar::acos_me)
      .def("asin", &CppADScalar::asin_me)
      .def("atan", &CppADScalar::atan_me)
      .def("cos", &CppADScalar::cos_me)
      .def("cosh", &CppADScalar::cosh_me)
      .def("exp", &CppADScalar::exp_me)
      .def("fabs", &CppADScalar::fabs_me)
      .def("log", &CppADScalar::log_me)
      .def("sin", &CppADScalar::sin_me)
      .def("sign", &CppADScalar::sign_me)
      .def("sinh", &CppADScalar::sinh_me)
      .def("sqrt", &CppADScalar::sqrt_me)
      .def("tan", &CppADScalar::tan_me)
      .def("tanh", &CppADScalar::tanh_me)
      .def("asinh", &CppADScalar::asinh_me)
      .def("acosh", &CppADScalar::acosh_me)
      .def("atanh", &CppADScalar::atanh_me)
      .def("erf", &CppADScalar::erf_me)
      .def("expm1", &CppADScalar::expm1_me)
      .def("log1p", &CppADScalar::log1p_me)
      .def("__repr__", [](const CppADScalar& s) {
        return "<" + std::to_string(CppAD::Value(s)) + ">";
      });

  py::class_<CGScalar>(m, "CGScalar")
      .def(py::init([](double t) { return CGScalar(t); }))
      .def(py::init([](const CGScalar& scalar) { return CGScalar(scalar); }))
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
           [](const CGScalar& s, const CGScalar& exponent) {
             return CppAD::pow(s, exponent);
           })
      .def("__pow__", [](const CGScalar& s,
                         double exponent) { return CppAD::pow(s, exponent); })
      .def("abs", &CGScalar::abs_me)
      .def("acos", &CGScalar::acos_me)
      .def("asin", &CGScalar::asin_me)
      .def("atan", &CGScalar::atan_me)
      .def("cos", &CGScalar::cos_me)
      .def("cosh", &CGScalar::cosh_me)
      .def("exp", &CGScalar::exp_me)
      .def("fabs", &CGScalar::fabs_me)
      .def("log", &CGScalar::log_me)
      .def("sin", &CGScalar::sin_me)
      .def("sign", &CGScalar::sign_me)
      .def("sinh", &CGScalar::sinh_me)
      .def("sqrt", &CGScalar::sqrt_me)
      .def("tan", &CGScalar::tan_me)
      .def("tanh", &CGScalar::tanh_me)
      .def("asinh", &CGScalar::asinh_me)
      .def("acosh", &CGScalar::acosh_me)
      .def("atanh", &CGScalar::atanh_me)
      .def("erf", &CGScalar::erf_me)
      .def("expm1", &CGScalar::expm1_me)
      .def("log1p", &CGScalar::log1p_me)
      .def("__repr__", [](const CGScalar& s) {
        return "<" + std::to_string(CppAD::Value(s).getValue()) + ">";
      });
  ;

  // For CppADScalar
  py::class_<CppAD::ADFun<double>>(m, "CppADFunction")
      .def(py::init([](const ADVector& x, const ADVector& y) {
        return CppAD::ADFun<double>(x, y);
      }))
      .def(
          "forward",
          [](CppAD::ADFun<double>& f, const std::vector<double>& xq, size_t q) {
            return f.Forward(q, xq);
          },
          "Evaluates the forward pass of the function", py::arg("x"),
          py::arg("order") = 0)
      .def(
          "jacobian",
          [](CppAD::ADFun<double>& f, const std::vector<double>& x) {
            return f.Jacobian(x);
          },
          "Evaluates the Jacobian of the function")
      .def("to_json", &CppAD::ADFun<double>::to_json,
           "Represents the traced function by a JSON string");

  m.def("independent", [](ADVector& x) { CppAD::Independent(x); });

  // For CGScalar
  py::class_<std::shared_ptr<CppAD::ADFun<CppAD::cg::CG<double>>>>(m,
                                                                   "CGFunction")
      .def(py::init([](const CGVector& x, const CGVector& y) {
        return std::make_shared<CppAD::ADFun<CppAD::cg::CG<double>>>(x, y);
      }))
      //      .def(
      //          "forward",
      //          [](std::shared_ptr<CppAD::ADFun<CppAD::cg::CG<double>>>& f,
      //             const std::vector<double>& xq,
      //             size_t q) { return f->Forward(q, xq); },
      //          "Evaluates the forward pass of the function", py::arg("x"),
      //          py::arg("order") = 0)
      //      .def(
      //          "jacobian",
      //          [](CppAD::ADFun<CppAD::cg::CG<double>>& f,
      //             const std::vector<double>& x) { return f.Jacobian(x); },
      //          "Evaluates the Jacobian of the function")
      //      .def("to_json", &CppAD::ADFun<CppAD::cg::CG<double>>::to_json,
      //           "Represents the traced function by a JSON string")
      ;

  m.def("independent", [](CGVector& x) { CppAD::Independent(x); });

  py::class_<autogen::GeneratedLightWeight<double>>(m, "Autogen")
            .def(py::init<const std::string&, std::shared_ptr<ADFun>>())
//      .def(py::init([](const std::string& name, std::shared_ptr<ADFun> fun) {
//        return autogen::GeneratedLightWeight<double>(name, fun);
//      }))
        ;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
