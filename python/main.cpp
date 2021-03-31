#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "autogen/autogen.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using CppADScalar = typename CppAD::AD<double>;
using ADVector = std::vector<CppAD::AD<double>>;

PYBIND11_MODULE(autogen_python, m) {
  m.doc() = R"pbdoc(
        Autogen python plugin
        -----------------------

        .. currentmodule:: autogen_python

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  py::class_<CppADScalar>(m, "CppADScalar")
      .def(py::init<double>())
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
           [](CppADScalar& s, int exponent) { return CppAD::pow(s, exponent); })
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
      .def("to_base", [](CppADScalar& s) { return CppAD::Value(s); });

  py::class_<CppAD::ADFun<double>>(m, "ADFun")
      .def(py::init<ADVector, ADVector>())
//      .def("Forward", &CppAD::ADFun<double>::Forward<ADVector>) // TODO:fix
//      .def("Jacobian", &CppAD::ADFun<double>::Jacobian<ADVector>) // TODO:fix
      ;

//  m.def("Independent", &CppAD::Independent<std::vector<CppAD::AD<double>>>); // TODO:fix

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
