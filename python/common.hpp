

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename Scalar>
py::class_<Scalar> expose_scalar(py::handle m, const char* name) {
  return py::class_<Scalar>(m, name)
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