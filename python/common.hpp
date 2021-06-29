#pragma once

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "autogen/autogen.hpp"

namespace py = pybind11;

using ADScalar = typename CppAD::AD<double>;
using CGScalar = typename CppAD::cg::CG<double>;
using ADCGScalar = typename CppAD::AD<CGScalar>;

using ADVector = std::vector<ADScalar>;
using ADCGVector = std::vector<ADCGScalar>;

using ADFun = typename CppAD::ADFun<double>;
using ADCGFun = typename CppAD::ADFun<CGScalar>;

template <typename Scalar>
py::class_<Scalar, std::shared_ptr<Scalar>> expose_scalar(py::handle m,
                                                          const char* name) {
  return py::class_<Scalar, std::shared_ptr<Scalar>>(m, name)
      .def(py::init([](double t) {
        return Scalar(t);
      }))
      .def(py::init([](const Scalar& scalar) {
        return Scalar(scalar);
      }))
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

template <template <typename> typename Functor>
struct publish_function {
  Functor<ADScalar> functor_ad;
  Functor<ADCGScalar> functor_cg;

  void operator()(py::module& m, const char* name) {
    m.def(name, [this, name](const ADScalar& x) -> ADScalar {
      std::cout << "Retrieving ADScalar tape table...\n";
      ADScalar::tape_table[0] = reinterpret_cast<CppAD::local::ADTape<double>*>(
          py::get_shared_data("tape_table_ad"));
      std::cout << "Calling CppAD " << name << " with x = " << x << "\n";
      return functor_ad(x);
    });

    m.def(name, [this, name](const ADCGScalar& x) -> ADCGScalar {
      std::cout << "Retrieving ADCGScalar tape table...\n";
      ADCGScalar::tape_table[0] =
          reinterpret_cast<CppAD::local::ADTape<CGScalar>*>(
              py::get_shared_data("tape_table_adcg"));
      std::cout << "Calling CodeGen " << name << " with x = " << x << "\n";
      return functor_cg(x);
    });
  }
};