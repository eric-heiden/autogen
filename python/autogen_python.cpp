#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cppad/utility/thread_alloc.hpp>

#include "autogen/autogen.hpp"
#include "autogen/core/base.hpp"
#include "common.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace autogen;

PYBIND11_MAKE_OPAQUE(ADVector);
PYBIND11_MAKE_OPAQUE(ADCGVector);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADFun>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ADCGFun>);

template <typename Scalar>
void expose_where_gt(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const Scalar& x, const Scalar& y, const Scalar& if_true,
         const Scalar& if_false) {
        return autogen::where_gt(x, y, if_true, if_false);
      },
      py::arg("x"), py::arg("y"), py::arg("if_true"), py::arg("if_false"));
}
template <typename Scalar>
void expose_where_lt(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const Scalar& x, const Scalar& y, const Scalar& if_true,
         const Scalar& if_false) {
        return autogen::where_lt(x, y, if_true, if_false);
      },
      py::arg("x"), py::arg("y"), py::arg("if_true"), py::arg("if_false"));
}
template <typename Scalar>
void expose_where_ge(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const Scalar& x, const Scalar& y, const Scalar& if_true,
         const Scalar& if_false) {
        return autogen::where_ge(x, y, if_true, if_false);
      },
      py::arg("x"), py::arg("y"), py::arg("if_true"), py::arg("if_false"));
}
template <typename Scalar>
void expose_where_le(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const Scalar& x, const Scalar& y, const Scalar& if_true,
         const Scalar& if_false) {
        return autogen::where_le(x, y, if_true, if_false);
      },
      py::arg("x"), py::arg("y"), py::arg("if_true"), py::arg("if_false"));
}
template <typename Scalar>
void expose_where_eq(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const Scalar& x, const Scalar& y, const Scalar& if_true,
         const Scalar& if_false) {
        return autogen::where_eq(x, y, if_true, if_false);
      },
      py::arg("x"), py::arg("y"), py::arg("if_true"), py::arg("if_false"));
}

PYBIND11_MODULE(_autogen, m) {
  m.doc() = R"pbdoc(
        Autogen python plugin
        -----------------------

        .. currentmodule:: autogen

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  global_scope_ = new Scope;
  py::set_shared_data("scope", global_scope_);
  // std::cout << "Welcome to autogen!\n";

  py::bind_vector<ADVector>(m, "ADVector");
  py::bind_vector<ADCGVector>(m, "ADCGVector");

  py::enum_<ScalarType>(m, "Mode")
      .value("DOUBLE", ScalarType::SCALAR_DOUBLE)
      .value("CPPAD", ScalarType::SCALAR_CPPAD)
      .value("CODEGEN", ScalarType::SCALAR_CODEGEN)
      .export_values();

  m.def("get_mode", []() { return get_scope()->mode; });
  m.def("set_mode", [](const ScalarType& mode) { get_scope()->mode = mode; });

  expose_scalar<ADScalar>(m, "ADScalar").def("__repr__", [](const ADScalar& s) {
    return "ad<" + std::to_string(CppAD::Value(CppAD::Var2Par(s))) + ">";
  });
  expose_scalar<ADCGScalar>(m, "ADCGScalar")
      .def("__repr__", [](const ADCGScalar& s) {
        return "adcg<" +
               std::to_string(CppAD::Value(CppAD::Var2Par(s)).getValue()) + ">";
      });

  // py::implicitly_convertible<double, ADScalar>();
  // py::implicitly_convertible<int, ADScalar>();
  // py::implicitly_convertible<double, ADCGScalar>();
  // py::implicitly_convertible<int, ADCGScalar>();

  m.def("scalar_name", []() {
    switch (get_scope()->mode) {
      case SCALAR_CPPAD:
        return "AD";
      case SCALAR_CODEGEN:
        return "ADCG";
      case SCALAR_DOUBLE:
      default:
        return "double";
    }
  });

  m.def("to_double", [](double d) -> double { return d; });
  m.def("to_double", [](const ADScalar& s) -> double {
    return CppAD::Value(CppAD::Var2Par(s));
  });
  m.def("to_double", [](const ADCGScalar& s) -> double {
    return CppAD::Value(CppAD::Var2Par(s)).getValue();
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
      .def_property_readonly(
          "input_dim",
          [](const std::shared_ptr<ADFun>& fun) { return fun->Domain(); })
      .def_property_readonly(
          "output_dim",
          [](const std::shared_ptr<ADFun>& fun) { return fun->Range(); })
      .def("optimize",
           [](std::shared_ptr<ADFun>& fun) { return fun->optimize(); });
  // .def("to_json", &ADFun::to_json,
  //      "Represents the traced function by a JSON string");

  m.def("independent", [](ADVector& x) {
    CppAD::Independent(x);
    // XXX save tape table for thread 0
    py::set_shared_data("tape_table_ad", ADScalar::tape_table[0]);
    // py::set_shared_data("traces", &CodeGenData<BaseScalar>::traces);
    // py::set_shared_data("is_dry_run",
    // &CodeGenData<BaseScalar>::is_dry_run);
    // py::set_shared_data("call_hierarchy",
    // &CodeGenData<BaseScalar>::call_hierarchy);
    // py::set_shared_data("invocation_order",
    // CodeGenData<BaseScalar>::invocation_order);
  });

  // For ADCGScalar
  py::class_<std::shared_ptr<ADCGFun>>(m, "ADCGFun")
      .def(py::init([](const ADCGVector& x, const ADCGVector& y) {
        return std::make_shared<ADCGFun>(x, y);
      }))
      .def_property_readonly(
          "input_dim",
          [](const std::shared_ptr<ADCGFun>& fun) { return fun->Domain(); })
      .def_property_readonly(
          "output_dim",
          [](const std::shared_ptr<ADCGFun>& fun) { return fun->Range(); })
      .def("optimize",
           [](std::shared_ptr<ADCGFun>& fun) { return fun->optimize(); });

  m.def("independent", [](ADCGVector& x) {
    CppAD::Independent(x);
    // XXX save tape table for thread 0
    py::set_shared_data("tape_table_adcg", ADCGScalar::tape_table[0]);
    py::set_shared_data("tape_id_table", ADCGScalar::tape_id_table);
    py::set_shared_data("atomic_index_infos", ADCGScalar::atomic_index_infos);
    py::set_shared_data("traces", CodeGenData<BaseScalar>::traces);
    py::set_shared_data("is_dry_run", &CodeGenData<BaseScalar>::is_dry_run);
    py::set_shared_data("call_hierarchy",
                        &CodeGenData<BaseScalar>::call_hierarchy);

    std::cout << "when calling independent: ";
    print_invocation_order();

    py::set_shared_data("invocation_order",
                        CodeGenData<BaseScalar>::invocation_order);
    std::cout << "ADCG Atomic index infos has "
              << ADCGScalar::atomic_index_infos->size() << " entries.\n";
  });

  py::class_<autogen::GeneratedCppAD>(m, "GeneratedCppAD")
      .def(py::init<std::shared_ptr<ADFun>>())
      .def(py::init([](std::shared_ptr<ADFun> fun) {
        return autogen::GeneratedCppAD(fun);
      }))
      .def(
          "__call__",
          [](autogen::GeneratedCppAD& gen,
             const std::vector<BaseScalar>& input) {
            std::vector<BaseScalar> output;
            gen(input, output);
            return output;
          },
          "Evaluates the zero-order forward pass of the function",
          py::is_operator())
      .def(
          "__call__",
          [](autogen::GeneratedCppAD& gen,
             const std::vector<std::vector<BaseScalar>>& local_inputs,
             const std::vector<BaseScalar>& global_input) {
            std::vector<std::vector<BaseScalar>> outputs;
            gen(local_inputs, outputs, global_input);
            return outputs;
          },
          "Evaluates the zero-order forward pass of the function",
          py::is_operator())
      .def(
          "jacobian",
          [](autogen::GeneratedCppAD& gen,
             const std::vector<BaseScalar>& input) {
            std::vector<BaseScalar> output;
            gen.jacobian(input, output);
            return output;
          },
          "Evaluates the Jacobian of the function")
      .def(
          "jacobian",
          [](autogen::GeneratedCppAD& gen,
             const std::vector<std::vector<BaseScalar>>& local_inputs,
             const std::vector<BaseScalar>& global_input) {
            std::vector<std::vector<BaseScalar>> outputs;
            gen.jacobian(local_inputs, outputs, global_input);
            return outputs;
          },
          "Evaluates the Jacobian of the function")
      .def_property_readonly("local_input_dim",
                             &autogen::GeneratedCppAD::local_input_dim)
      .def_property_readonly("output_dim", &autogen::GeneratedCppAD::output_dim)
      .def_property("global_input_dim",
                    &autogen::GeneratedCppAD::global_input_dim,
                    &autogen::GeneratedCppAD::set_global_input_dim);

  py::class_<autogen::GeneratedCodeGen,
             std::shared_ptr<autogen::GeneratedCodeGen>>(m, "GeneratedCodeGen")
      .def(py::init<const std::string&, std::shared_ptr<ADCGFun>>())
      .def(
          "__call__",
          [](autogen::GeneratedCodeGen& gen,
             const std::vector<BaseScalar>& input) {
            std::vector<BaseScalar> output;
            if (input.size() != gen.input_dim()) {
              throw std::runtime_error(
                  "Input vector to function " + gen.library_name() +
                  " has to be of dimension " + std::to_string(gen.input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(input.size()) + ".");
            }
            gen(input, output);
            return output;
          },
          "Evaluates the zero-order forward pass of the function",
          py::is_operator())
      .def(
          "__call__",
          [](autogen::GeneratedCodeGen& gen,
             const std::vector<std::vector<BaseScalar>>& local_inputs,
             const std::vector<BaseScalar>& global_input) {
            std::vector<std::vector<BaseScalar>> outputs;
            if (!local_inputs.empty() &&
                local_inputs[0].size() != gen.local_input_dim()) {
              throw std::runtime_error(
                  "Local input vector to function " + gen.library_name() +
                  " has to be of dimension " +
                  std::to_string(gen.local_input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(local_inputs[0].size()) + ".");
            }
            if (global_input.size() != gen.global_input_dim()) {
              throw std::runtime_error(
                  "Global input vector to function " + gen.library_name() +
                  " has to be of dimension " +
                  std::to_string(gen.global_input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(global_input.size()) + ".");
            }
            gen(local_inputs, outputs, global_input);
            return outputs;
          },
          "Evaluates the zero-order forward pass of the function",
          py::is_operator())
      .def(
          "jacobian",
          [](autogen::GeneratedCodeGen& gen,
             const std::vector<BaseScalar>& input) {
            std::vector<BaseScalar> output;
            if (input.size() != gen.input_dim()) {
              throw std::runtime_error(
                  "Input vector to function " + gen.library_name() +
                  " has to be of dimension " + std::to_string(gen.input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(input.size()) + ".");
            }
            gen.jacobian(input, output);
            return output;
          },
          "Evaluates the Jacobian of the function")
      .def(
          "jacobian",
          [](autogen::GeneratedCodeGen& gen,
             const std::vector<std::vector<BaseScalar>>& local_inputs,
             const std::vector<BaseScalar>& global_input) {
            std::vector<std::vector<BaseScalar>> outputs;
            if (!local_inputs.empty() &&
                local_inputs[0].size() != gen.local_input_dim()) {
              throw std::runtime_error(
                  "Local input vector to function " + gen.library_name() +
                  " has to be of dimension " +
                  std::to_string(gen.local_input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(local_inputs[0].size()) + ".");
            }
            if (global_input.size() != gen.global_input_dim()) {
              throw std::runtime_error(
                  "Global input vector to function " + gen.library_name() +
                  " has to be of dimension " +
                  std::to_string(gen.global_input_dim()) +
                  ". Provided was a vector of dimension " +
                  std::to_string(global_input.size()) + ".");
            }
            gen.jacobian(local_inputs, outputs, global_input);
            return outputs;
          },
          "Evaluates the Jacobian of the function")
      .def("compile_cpu", &autogen::GeneratedCodeGen::compile_cpu,
           "Compile to a CPU-bound shared library",
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("compile_cuda", &autogen::GeneratedCodeGen::compile_cuda,
           "Compile to a GPU-bound shared library",
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def_property_readonly("local_input_dim",
                             &autogen::GeneratedCodeGen::local_input_dim)
      .def_property_readonly("output_dim",
                             &autogen::GeneratedCodeGen::output_dim)
      .def_property_readonly("input_dim", &autogen::GeneratedCodeGen::input_dim)
      .def_property("global_input_dim",
                    &autogen::GeneratedCodeGen::global_input_dim,
                    &autogen::GeneratedCodeGen::set_global_input_dim)
      .def_property_readonly("is_compiled",
                             &autogen::GeneratedCodeGen::is_compiled)
      .def_readwrite("debug_mode", &autogen::GeneratedCodeGen::debug_mode)
      .def("discard_library", &autogen::GeneratedCodeGen::discard_library)
      .def_property("library_name", &autogen::GeneratedCodeGen::library_name,
                    &autogen::GeneratedCodeGen::load_precompiled_library);

  py::class_<CodeGenData<BaseScalar>>(m, "CodeGenData")
      .def_static("clear", &CodeGenData<BaseScalar>::clear)
      .def_static("has_trace",
                  [](const std::string& name) {
                    return CodeGenData<BaseScalar>::traces->find(name) !=
                           CodeGenData<BaseScalar>::traces->end();
                  })
      .def_static("update_call_hierarchy",
                  [](const std::string& name) {
                    auto& order = *CodeGenData<BaseScalar>::invocation_order;
                    if (!order.empty()) {
                      // the current function is called by another function,
                      // hence update the call hierarchy
                      const std::string& parent = order.back();
                      auto& hierarchy = CodeGenData<BaseScalar>::call_hierarchy;
                      if (hierarchy.find(parent) == hierarchy.end()) {
                        hierarchy[parent] = std::vector<std::string>();
                      }
                      hierarchy[parent].push_back(name);
                    }
                    order.push_back(name);
                  })
      .def_static("set_dry_run",
                  [](bool dry_run) {
                    CodeGenData<BaseScalar>::is_dry_run = dry_run;
                    std::cout << "Setting dry run to " << std::boolalpha
                              << CodeGenData<BaseScalar>::is_dry_run << "\n";
                  })
      .def_readwrite_static("invocation_order",
                            &CodeGenData<BaseScalar>::invocation_order)
      .def_readwrite_static("call_hierarchy",
                            &CodeGenData<BaseScalar>::call_hierarchy)
      .def_static(
          "register_trace",
          [](const std::string& name, const std::shared_ptr<ADCGFun>& tape) {
            using CGAtomicFunBridge =
                typename CppAD::cg::CGAtomicFunBridge<BaseScalar>;
            FunctionTrace<BaseScalar>& trace =
                (*CodeGenData<BaseScalar>::traces)[name];
            std::cout << "Adding trace for atomic function \"" << trace.name
                      << "\"...\n";
            trace.tape = tape;
            trace.bridge = new CGAtomicFunBridge(name, *(trace.tape), true);
            trace.input_dim = tape->Domain();
            trace.output_dim = tape->Range();
          })
      .def_static("call_bridge", [](const std::string& name,
                                    const ADCGVector& input) {
        if (CodeGenData<BaseScalar>::traces->find(name) ==
            CodeGenData<BaseScalar>::traces->end()) {
          throw std::runtime_error("Could not find trace with name \"" + name +
                                   "\" while attempting to call the "
                                   "corresponding function bridge.");
        }
        FunctionTrace<BaseScalar>& trace =
            (*CodeGenData<BaseScalar>::traces)[name];
        ADCGVector output(trace.output_dim);
        (*(trace.bridge))(input, output);
        return output;
      });

  expose_where_gt<BaseScalar>(m, "where_gt");
  expose_where_gt<ADScalar>(m, "where_gt");
  expose_where_gt<ADCGScalar>(m, "where_gt");

  expose_where_lt<BaseScalar>(m, "where_lt");
  expose_where_lt<ADScalar>(m, "where_lt");
  expose_where_lt<ADCGScalar>(m, "where_lt");

  expose_where_ge<BaseScalar>(m, "where_ge");
  expose_where_ge<ADScalar>(m, "where_ge");
  expose_where_ge<ADCGScalar>(m, "where_ge");

  expose_where_le<BaseScalar>(m, "where_le");
  expose_where_le<ADScalar>(m, "where_le");
  expose_where_le<ADCGScalar>(m, "where_le");

  expose_where_eq<BaseScalar>(m, "where_eq");
  expose_where_eq<ADScalar>(m, "where_eq");
  expose_where_eq<ADCGScalar>(m, "where_eq");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
