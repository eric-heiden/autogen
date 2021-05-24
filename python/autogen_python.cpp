#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "autogen/autogen.hpp"
#include "common.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(ADVector);
PYBIND11_MAKE_OPAQUE(ADCGVector);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADFun>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADCGFun>);

template <typename Scalar>
struct my_traceable_function2 {
  Scalar operator()(const Scalar &x) const {
    using std::cos;
    return cos(x) * x;
  }
};

PYBIND11_MODULE(_autogen, m) {
  m.doc() = R"pbdoc(
        Autogen python plugin
        -----------------------

        .. currentmodule:: autogen

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  py::bind_vector<ADVector>(m, "ADVector");
  py::bind_vector<ADCGVector>(m, "ADCGVector");

  expose_scalar<ADScalar>(m, "ADScalar")
      .def("__repr__", [](const ADScalar& s) {
        return "ad<" + std::to_string(CppAD::Value(CppAD::Var2Par(s))) + ">";
      })
      .def("value", [](const ADScalar& s) {
        return CppAD::Value(CppAD::Var2Par(s));
      })
      ;
  expose_scalar<ADCGScalar>(m, "ADCGScalar")
      .def("__repr__", [](const ADCGScalar& s) {
        return "adcg<" +
               std::to_string(CppAD::Value(CppAD::Var2Par(s)).getValue()) + ">";
      });

  // For CppADScalar
  py::class_<std::shared_ptr<ADFun>>(m, "ADFun")
      .def(py::init([](const ADVector& x, const ADVector& y) {
        return std::make_shared<ADFun>(x, y);
      }))
      .def(
          "forward",
          [](std::shared_ptr<ADFun>& f, const std::vector<double>& xq,
             size_t q) { return f->Forward(q, xq); },
          "Evaluates the forward pass of the function", py::arg("x"),
          py::arg("order") = 0)
      .def(
          "jacobian",
          [](std::shared_ptr<ADFun>& f, const std::vector<double>& x) {
            return f->Jacobian(x);
          },
          "Evaluates the Jacobian of the function")
      //      .def("to_json", &ADFun::to_json,
      //           "Represents the traced function by a JSON string")
      ;

  m.def("independent", [](ADVector& x) { CppAD::Independent(x); });

  // For ADCGScalar
  py::class_<std::shared_ptr<ADCGFun>>(m, "ADCGFun")
      .def(py::init([](const ADCGVector& x, const ADCGVector& y) {
        return std::make_shared<ADCGFun>(x, y);
      }));

  m.def("independent", [](ADCGVector& x) { CppAD::Independent(x); });

  // py::class_<autogen::GeneratedLightWeight<double>>(m, "Autogen")
  //     .def(py::init<const std::string&, std::shared_ptr<ADFun>>())
  //      .def(py::init([](const std::string& name, std::shared_ptr<ADFun>
  //      fun) {
  //        return autogen::GeneratedLightWeight<double>(name, fun);
  //      }))
  //  ;

  py::class_<autogen::GeneratedCppAD>(m, "GeneratedCppAD")
      .def(py::init<std::shared_ptr<ADFun>>())
      .def(py::init([](std::shared_ptr<ADFun> fun) {
        return autogen::GeneratedCppAD(fun);
      }))
      .def(
          "forward",
          [](autogen::GeneratedCppAD& gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen(input, output);
            return output;
          },
          "Evaluates the zero-order forward pass of the function")
      .def(
          "jacobian",
          [](autogen::GeneratedCppAD& gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen.jacobian(input, output);
            return output;
          },
          "Evaluates the Jacobian of the function");;

  py::class_<autogen::GeneratedCodeGen,
             std::shared_ptr<autogen::GeneratedCodeGen>>(m, "GeneratedCodeGen")
      .def(py::init<const std::string&, std::shared_ptr<ADCGFun>>())
      .def(
          "forward",
          [](autogen::GeneratedCodeGen& gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen(input, output);
            return output;
          },
          "Evaluates the zero-order forward pass of the function")
      .def(
          "jacobian",
          [](autogen::GeneratedCodeGen& gen, const std::vector<double>& input) {
            std::vector<double> output;
            gen.jacobian(input, output);
            return output;
          },
          "Evaluates the Jacobian of the function")
      .def("compile_cpu", &autogen::GeneratedCodeGen::compile_cpu,
           "Compile to a CPU-bound shared library")
      .def("compile_cuda", &autogen::GeneratedCodeGen::compile_cuda,
           "Compile to a GPU-bound shared library");


  publish_function<my_traceable_function2>(m, "my_traceable_function2");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
