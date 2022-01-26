#pragma once

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "autogen/autogen.hpp"

namespace py = pybind11;
using namespace autogen;

#ifdef PYBIND11_NUMPY_OBJECT_DTYPE
PYBIND11_NUMPY_OBJECT_DTYPE(ADScalar);
PYBIND11_NUMPY_OBJECT_DTYPE(ADCGScalar);
#endif

struct Scope {
  GenerationMode mode{MODE_NUMERICAL};
};

static Scope* global_scope_ = nullptr;

Scope* get_scope() {
  if (!global_scope_) {
    global_scope_ = reinterpret_cast<Scope*>(py::get_shared_data("scope"));
  }
  return global_scope_;
}

template <typename Scalar>
py::class_<Scalar, std::shared_ptr<Scalar>> expose_scalar(py::handle m,
                                                          const char* name) {
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
      .def("atan2",
           [](const Scalar& y, const Scalar& x) { return CppAD::atan2(y, x); })
      .def("arctan2",
           [](const Scalar& y, const Scalar& x) { return CppAD::atan2(y, x); })           
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

template <typename Scalar>
static void retrieve_tape();

template <>
void retrieve_tape<ADScalar>() {
  // std::cout << "Retrieving ADScalar tape table...\n";
  ADScalar::tape_table[0] = reinterpret_cast<CppAD::local::ADTape<double>*>(
      py::get_shared_data("tape_table_ad"));
  // ADCGScalar::atomic_index_infos =
  //    std::shared_ptr<std::vector<CppAD::local::atomic_index_info>>(
  //    reinterpret_cast<std::vector<CppAD::local::atomic_index_info>*>(
  //         py::get_shared_data("atomic_index_infos_ad")));
  // autogen::CodeGenData::traces = *reinterpret_cast<
  //     std::map<std::string, autogen::FunctionTrace<autogen::BaseScalar>>*>(
  //     py::get_shared_data("traces"));
  // autogen::CodeGenData::is_dry_run =
  //     *reinterpret_cast<bool*>(py::get_shared_data("is_dry_run"));
  // autogen::CodeGenData::call_hierarchy =
  //     *reinterpret_cast<std::map<std::string, std::vector<std::string>>*>(
  //         py::get_shared_data("call_hierarchy"));
  // autogen::CodeGenData::invocation_order =
  //     reinterpret_cast<std::vector<std::string>*>(
  //         py::get_shared_data("invocation_order"));
  // std::cout << "AD restored Atomic index infos has "
  //           << ADCGScalar::atomic_index_infos->size() << " entries.\n";
}

void print_invocation_order() {
  std::cout << "Invocation order: ";
  for (const auto& s : autogen::CodeGenData::invocation_order()) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void retrieve_shared_ptr(T** target, const std::string& name) {
  // std::cout << "Retrieving shared pointer " << name << "...";
  *target = reinterpret_cast<T*>(py::get_shared_data(name));
  // if (!*target) {
  //   std::cout << "  NULL\n";
  // } else {
  //   std::cout << "  OK\n";
  // }
}

template <>
void retrieve_tape<ADCGScalar>() {
#ifdef DEBUG
  std::cout << "before retrieving tape ADCG restored Atomic index infos ("
            << ADCGScalar::atomic_index_infos << ") has "
            << ADCGScalar::atomic_index_infos->size() << " entries.\n";
  std::cout << "Retrieving ADCGScalar tape table...\n";
#endif
  ADCGScalar::tape_table = reinterpret_cast<CppAD::local::ADTape<CGScalar>**>(
      py::get_shared_data("tape_table_adcg"));
  ADCGScalar::tape_id_table =
      reinterpret_cast<CppAD::tape_id_t*>(py::get_shared_data("tape_id_table"));
  retrieve_shared_ptr(&ADCGScalar::atomic_index_infos, "atomic_index_infos");
  // autogen::CodeGenData::traces = reinterpret_cast<
  //     std::map<std::string, autogen::FunctionTrace<autogen::BaseScalar>>*>(
  //     py::get_shared_data("traces"));
  retrieve_shared_ptr(&autogen::CodeGenData::instance, "codegen_data");

  CppAD::thread_alloc::all_info =
      reinterpret_cast<CppAD::thread_alloc::thread_alloc_info**>(
          py::get_shared_data("thread_alloc_all_info"));
  CppAD::thread_alloc::zero_info =
      reinterpret_cast<CppAD::thread_alloc::thread_alloc_info*>(
          py::get_shared_data("thread_alloc_zero_info"));

  // autogen::CodeGenData::is_dry_run =
  //     *reinterpret_cast<bool*>(py::get_shared_data("is_dry_run"));
  // autogen::CodeGenData::call_hierarchy =
  //     *reinterpret_cast<std::map<std::string, std::vector<std::string>>*>(
  //         py::get_shared_data("call_hierarchy"));
  // std::cout << "before retrieving tape data:  ";
  // print_invocation_order();
  // autogen::CodeGenData::invocation_order =
  //     reinterpret_cast<std::vector<std::string>*>(
  //         py::get_shared_data("invocation_order"));
  // retrieve_shared_ptr(&autogen::CodeGenData::invocation_order,
  //                     "invocation_order");

#ifdef DEBUG
  std::cout << "after retrieved tape data:  ";
  print_invocation_order();
  std::cout << "ADCG restored Atomic index at "
            << ADCGScalar::atomic_index_infos << " infos has "
            << ADCGScalar::atomic_index_infos->size() << " entries.\n";
  std::cout << "Retrieved codegen_data ptr: " << autogen::CodeGenData::instance
            << std::endl;
#endif
}

void set_shared_data() {
  ADCGScalar::atomic_index_infos =
      new std::vector<CppAD::local::atomic_index_info>;
  py::set_shared_data("tape_table_adcg", ADCGScalar::tape_table);
  py::set_shared_data("tape_id_table", ADCGScalar::tape_id_table);
  py::set_shared_data("atomic_index_infos", ADCGScalar::atomic_index_infos);
  CodeGenData::create();
  py::set_shared_data("codegen_data", CodeGenData::instance);
  py::set_shared_data("thread_alloc_all_info", CppAD::thread_alloc::all_info);
  py::set_shared_data("thread_alloc_zero_info", CppAD::thread_alloc::zero_info);
  // py::set_shared_data("is_dry_run", &CodeGenData::is_dry_run);
  // py::set_shared_data("call_hierarchy", &CodeGenData::call_hierarchy);

  // py::set_shared_data("invocation_order", CodeGenData::invocation_order);
  std::cout << "ADCGScalar::atomic_index_infos  ptr: "
            << ADCGScalar::atomic_index_infos << std::endl;
  std::cout << "codegen_data  ptr: " << CodeGenData::instance << std::endl;
}

template <>
void retrieve_tape<double>() {
  // do nothing
}

template <typename Scalar>
static std::string scalar_name();

template <>
std::string scalar_name<ADScalar>() {
  return "AD";
}

template <>
std::string scalar_name<ADCGScalar>() {
  return "ADCG";
}

template <>
std::string scalar_name<double>() {
  return "double";
}

template <template <typename> typename Functor>
struct publish_function {
  Functor<double> functor_double;
  Functor<ADScalar> functor_ad;
  Functor<ADCGScalar> functor_cg;

  void operator()(py::module& m, const char* name) {
    m.def(name, [this, name](double x) {
      // std::cout << "Calling double " << name << " with x = " << x << "\n";
      return functor_double(x);
    });

    m.def(name, [this, name](const ADScalar& x) {
      // retrieve_tape<ADScalar>();
      // std::cout << "Calling CppAD " << name << " with x = " << x << "\n";
      return functor_ad(x);
    });

    m.def(name, [this, name](const ADCGScalar& x) {
      // retrieve_tape<ADCGScalar>();
      // std::cout << "Calling CodeGen " << name << " with x = " << x << "\n";
      return functor_cg(x);
    });
  }
};

template <template <typename> typename Functor>
struct publish_vec_function {
  Functor<double> functor_double;
  Functor<ADScalar> functor_ad;
  Functor<ADCGScalar> functor_cg;

  void operator()(py::module& m, const char* name) {
    m.def(name,
          [this, name](const std::vector<double>& x) -> std::vector<double> {
            // std::cout << "Calling double " << name << "\n";
            try {
              std::vector<double> output;
              functor_double(x, output);
              return output;
            } catch (const std::exception& ex) {
              std::cerr << "Error while calling function \"" << name << "\":\n"
                        << ex.what() << "\n";
              throw ex;
            }
          });

    m.def(
        name,
        [this, name](const std::vector<ADScalar>& x) -> std::vector<ADScalar> {
          // retrieve_tape<ADScalar>();
          // std::cout << "Calling CppAD " << name << "\n";
          try {
            std::vector<ADScalar> output;
            functor_ad(x, output);
            return output;
          } catch (const std::exception& ex) {
            std::cerr << "Error while calling function \"" << name << "\":\n"
                      << ex.what() << "\n";
            throw ex;
          }
        });

    m.def(name,
          [this,
           name](const std::vector<ADCGScalar>& x) -> std::vector<ADCGScalar> {
            // retrieve_tape<ADCGScalar>();
            // std::cout << "Calling CodeGen " << name << "\n";
            try {
              std::vector<ADCGScalar> output;
              functor_cg(x, output);
              return output;
            } catch (const std::exception& ex) {
              std::cerr << "Error while calling function \"" << name << "\":\n"
                        << ex.what() << "\n";
              throw ex;
            }
          });
  }
};

template <template <typename> typename Class, typename Scalar,
          template <typename> typename... Parents>
using PyClassPublishHandle =
    typename py::class_<Class<Scalar>, std::shared_ptr<Class<Scalar>>,
                        Parents<Scalar>...>;

template <template <typename> typename Class, typename Scalar,
          template <typename> typename... Parents>
struct ClassPublisher {
  virtual void operator()(
      PyClassPublishHandle<Class, Scalar, Parents...>& handle) const = 0;
};

template <template <typename> typename Class,
          template <typename> typename... Parents>
void publish_class(py::module& m, const std::string& name) {
  using HandleDouble =
      typename py::class_<Class<double>, std::shared_ptr<Class<double>>,
                          Parents<double>...>;
  using HandleAD =
      typename py::class_<Class<ADScalar>, std::shared_ptr<Class<ADScalar>>,
                          Parents<ADScalar>...>;
  using HandleADCG =
      typename py::class_<Class<ADCGScalar>, std::shared_ptr<Class<ADCGScalar>>,
                          Parents<ADCGScalar>...>;
  auto handle_double = HandleDouble(m, (name + "_double").c_str());
  auto pub_double = ClassPublisher<Class, double, Parents...>();
  (pub_double)(handle_double);
  auto handle_ad = HandleAD(m, (name + "_ad").c_str());
  auto pub_ad = ClassPublisher<Class, ADScalar, Parents...>();
  (pub_ad)(handle_ad);
  auto handle_adcg = HandleADCG(m, (name + "_adcg").c_str());
  auto pub_adcg = ClassPublisher<Class, ADCGScalar, Parents...>();
  (pub_adcg)(handle_adcg);
  auto handle_type =
      m.def(name.c_str(), [&m, &name](py::args args, const py::kwargs& kwargs) {
        switch (get_scope()->mode) {
          case MODE_CPPAD: {
            // retrieve_tape<ADScalar>();
            // std::cout << "returning CppAD\n";
            py::type type = py::type::of<Class<ADScalar>>();
            return type(*args, **kwargs);
          }
          case MODE_CODEGEN: {
            // retrieve_tape<ADCGScalar>();
            // std::cout << "returning CodeGen\n";
            py::type type = py::type::of<Class<ADCGScalar>>();
            return type(*args, **kwargs);
          }
          case MODE_NUMERICAL:
          default: {
            // std::cout << "returning double\n";
            py::type type = py::type::of<Class<double>>();
            return type(*args, **kwargs);
          }
        }
      });
}