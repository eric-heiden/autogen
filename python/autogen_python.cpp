#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "autogen/autogen.hpp"
#include "common.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using CppADScalar = typename CppAD::AD<double>;
using CGInnerScalar = typename CppAD::cg::CG<double>;
using CGScalar = typename CppAD::AD<CGInnerScalar>;
using ADVector = std::vector<CppADScalar>;
using CGVector = std::vector<CGScalar>;

using ADFun = typename CppAD::ADFun<CGInnerScalar>;

PYBIND11_MAKE_OPAQUE(ADVector);
PYBIND11_MAKE_OPAQUE(CGVector);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADFun>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<CppAD::ADFun<double>>);

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

  expose_scalar<CppADScalar>(m, "CppADScalar")
      .def("__repr__", [](const CppADScalar& s) {
        return "<" + std::to_string(CppAD::Value(CppAD::Var2Par(s))) + ">";
      });
  expose_scalar<CGScalar>(m, "CGScalar").def("__repr__", [](const CGScalar& s) {
    return "<" + std::to_string(CppAD::Value(CppAD::Var2Par(s)).getValue()) +
           ">";
  });

  // For CppADScalar
  py::class_<std::shared_ptr<CppAD::ADFun<double>>>(m, "CppADFunction")
      .def(py::init([](const ADVector& x, const ADVector& y) {
        return std::make_shared<CppAD::ADFun<double>>(x, y);
      }))
      .def(
          "forward",
          [](std::shared_ptr<CppAD::ADFun<double>>& f, const std::vector<double>& xq, size_t q) {
            return f->Forward(q, xq);
          },
          "Evaluates the forward pass of the function", py::arg("x"),
          py::arg("order") = 0)
      .def(
          "jacobian",
          [](std::shared_ptr<CppAD::ADFun<double>>& f, const std::vector<double>& x) {
            return f->Jacobian(x);
          },
          "Evaluates the Jacobian of the function")
//      .def("to_json", &CppAD::ADFun<double>::to_json,
//           "Represents the traced function by a JSON string")
           ;

  m.def("independent", [](ADVector& x) { CppAD::Independent(x); });

  // For CGScalar
//  py::class_<std::shared_ptr<ADFun>>(m, "CGFunction")
//      .def(py::init([](const CGVector& x, const CGVector& y) {
//        return std::make_shared<ADFun>(x, y);
//      }))
//      .def(
//          "forward",
//          [](std::shared_ptr<ADFun> f, const std::vector<double>& xq,
//             size_t q) { return f->Forward(q, xq); },
//          "Evaluates the forward pass of the function", py::arg("x"),
//          py::arg("order") = 0)
//      .def(
//          "jacobian",
//          [](std::shared_ptr<ADFun> f, const std::vector<double>& x) {
//            return f->Jacobian(x);
//          },
//          "Evaluates the Jacobian of the function")
//      .def("to_json", &ADFun::to_json,
//           "Represents the traced function by a JSON string");

  // m.def("independent", [](CGVector& x) { CppAD::Independent(x); });

  // py::class_<autogen::GeneratedLightWeight<double>>(m, "Autogen")
  //     .def(py::init<const std::string&, std::shared_ptr<ADFun>>())
  //      .def(py::init([](const std::string& name, std::shared_ptr<ADFun>
  //      fun) {
  //        return autogen::GeneratedLightWeight<double>(name, fun);
  //      }))
  //  ;

   py::class_<autogen::GeneratedCppAD>(m, "GeneratedCppAD")
      .def(py::init<std::shared_ptr<CppAD::ADFun<double>>>())
      .def(py::init([](std::shared_ptr<CppAD::ADFun<double>> fun) {
        return autogen::GeneratedCppAD(fun);
      }))
      .def(
          "forward",
          [](autogen::GeneratedCppAD gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen(input, output);
            return output; })
      .def(
          "jacobian",
          [](autogen::GeneratedCppAD gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen.jacobian(input, output);
            return output;
          },
          "Evaluates the Jacobian of the function")
  ;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
