#include "autogen/cg/target/legacy_c_target.h"

#include "autogen/utils/system.h"

namespace autogen {

void LegacyCTarget::forward(const std::vector<BaseScalar> &input,
                            std::vector<BaseScalar> &output) {
  assert(!library_name_.empty());
  auto model = get_cpu_model();
  model->ForwardZero(input, output);
}

void LegacyCTarget::forward(
    const std::vector<std::vector<BaseScalar>> &local_inputs,
    std::vector<std::vector<BaseScalar>> &outputs,
    const std::vector<BaseScalar> &global_input) {
  outputs.resize(local_inputs.size());
  assert(!library_name_.empty());
  for (auto &o : outputs) {
    o.resize(this->output_dim());
  }
  int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
  for (int i = 0; i < num_tasks; ++i) {
    if (global_input.empty()) {
      auto model = get_cpu_model();
      model->ForwardZero(local_inputs[i], outputs[i]);
    } else {
      static thread_local std::vector<BaseScalar> input;
      input = global_input;
      input.resize(global_input.size() + local_inputs[0].size());
      for (size_t j = 0; j < local_inputs[i].size(); ++j) {
        input[j + global_input.size()] = local_inputs[i][j];
      }
      auto model = get_cpu_model();
      model->ForwardZero(input, outputs[i]);
    }
  }
}

void LegacyCTarget::jacobian(const std::vector<BaseScalar> &input,
                             std::vector<BaseScalar> &output) {
  auto model = get_cpu_model();
  model->Jacobian(input, output);
}

void LegacyCTarget::jacobian(
    const std::vector<std::vector<BaseScalar>> &local_inputs,
    std::vector<std::vector<BaseScalar>> &outputs,
    const std::vector<BaseScalar> &global_input) {
  outputs.resize(local_inputs.size());
  assert(!library_name_.empty());
  for (auto &o : outputs) {
    o.resize(this->input_dim() * this->output_dim());
  }
  int num_tasks = static_cast<int>(local_inputs.size());
#pragma omp parallel for
  for (int i = 0; i < num_tasks; ++i) {
    if (global_input.empty()) {
      auto model = get_cpu_model();
      // model->ForwardZero(local_inputs[i], outputs[i]);
      model->Jacobian(local_inputs[i], outputs[i]);
    } else {
      static thread_local std::vector<BaseScalar> input;
      if (input.empty()) {
        input.resize(global_input.size());
        input.insert(input.begin(), global_input.begin(), global_input.end());
      }
      for (size_t j = 0; j < local_inputs[i].size(); ++j) {
        input[j + global_input.size()] = local_inputs[i][j];
      }
      auto model = get_cpu_model();
      model->Jacobian(input, outputs[i]);
    }
  }
}

typename LegacyCTarget::GenericModelPtr LegacyCTarget::get_cpu_model() const {
  if (!cpu_library_) {
    cpu_library_loading_mutex_.lock();
    cpu_library_ = std::make_shared<DynamicLib>(library_name_ + library_ext_);
    std::set<std::string> model_names = cpu_library_->getModelNames();
    std::cout << "Successfully loaded CPU library "
              << library_name_ + library_ext_ << std::endl;
    for (auto &name : model_names) {
      std::cout << "  Found model " << name << std::endl;
    }
    // load and wire up atomic functions in this library
    const auto &order = *CodeGenData::invocation_order;
    const auto &hierarchy = CodeGenData::call_hierarchy;
    cpu_models_[this->name()] =
        GenericModelPtr(cpu_library_->model(this->name()).release());
    if (!cpu_models_[this->name()]) {
      throw std::runtime_error("Failed to load model from library " +
                               library_name_ + library_ext_);
    }
    // atomic functions to be added
    typedef std::pair<std::string, std::string> ParentChild;
    std::set<ParentChild> remaining_atomics;
    for (const std::string &s :
         cpu_models_[this->name()]->getAtomicFunctionNames()) {
      remaining_atomics.insert(std::make_pair(this->name(), s));
    }
    while (!remaining_atomics.empty()) {
      ParentChild member = *(remaining_atomics.begin());
      const std::string &parent = member.first;
      const std::string &atomic_name = member.second;
      remaining_atomics.erase(remaining_atomics.begin());
      if (cpu_models_.find(atomic_name) == cpu_models_.end()) {
        std::cout << "  Adding atomic function " << atomic_name << std::endl;
        cpu_models_[atomic_name] =
            GenericModelPtr(cpu_library_->model(atomic_name).release());
        for (const std::string &s :
             cpu_models_[atomic_name]->getAtomicFunctionNames()) {
          remaining_atomics.insert(std::make_pair(atomic_name, s));
        }
      }
      auto &atomic_model = cpu_models_[atomic_name];
      cpu_models_[parent]->addAtomicFunction(atomic_model->asAtomic());
    }

    std::cout << "Loaded compiled model \"" << this->name() << "\" from \""
              << library_name_ << "\".\n";
    cpu_library_loading_mutex_.unlock();
  }
  return cpu_models_[this->name()];
}

void LegacyCTarget::set_compiler_clang(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("clang");
  }
  cpu_compiler_ = std::make_shared<ClangCompiler>(compiler_path);
  for (const auto &flag : compile_flags) {
    cpu_compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    cpu_compiler_->addCompileLibFlag(flag);
  }
}

void LegacyCTarget::set_compiler_gcc(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("gcc");
  }
  cpu_compiler_ = std::make_shared<GccCompiler>(compiler_path);
  for (const auto &flag : compile_flags) {
    cpu_compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    cpu_compiler_->addCompileLibFlag(flag);
  }
}

void LegacyCTarget::set_compiler_msvc(
    std::string compiler_path, std::string linker_path,
    const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("cl.exe");
  }
  if (linker_path.empty()) {
    linker_path = autogen::find_exe("link.exe");
  }
  cpu_compiler_ = std::make_shared<MsvcCompiler>(compiler_path, linker_path);
  for (const auto &flag : compile_flags) {
    cpu_compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    cpu_compiler_->addCompileLibFlag(flag);
  }
}

bool LegacyCTarget::generate_code_() {
  using namespace CppAD;
  using namespace CppAD::cg;

  ModelCSourceGen<BaseScalar> main_source_gen(*(this->main_trace().tape),
                                              this->name());
  main_source_gen.setCreateForwardZero(generate_forward_);
  main_source_gen.setCreateJacobian(generate_jacobian_);
  libcgen_ =
      std::make_shared<ModelLibraryCSourceGen<BaseScalar>>(main_source_gen);
  // reverse order of invocation to first generate code for innermost
  // functions
  const auto &order = *CodeGenData::invocation_order;
  std::list<ModelCSourceGen<BaseScalar> *> models;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    FunctionTrace &trace = (*CodeGenData::traces)[*it];
    // trace.tape->optimize();
    auto *source_gen = new ModelCSourceGen<BaseScalar>(*(trace.tape), *it);
    source_gen->setCreateForwardZero(generate_forward_);
    // source_gen->setCreateSparseJacobian(generate_jacobian_);
    // source_gen->setCreateJacobian(generate_jacobian_);
    source_gen->setCreateForwardOne(generate_jacobian_);
    source_gen->setCreateReverseOne(generate_jacobian_);
    models.push_back(source_gen);
    // we need a stable reference
    libcgen_->addModel(*(models.back()));
  }
  libcgen_->setVerbose(true);

  SaveFilesModelLibraryProcessor<BaseScalar> psave(*libcgen_);
  psave.saveSourcesTo(sources_folder_);
  return true;
}

bool LegacyCTarget::compile_() {
  using namespace CppAD;
  using namespace CppAD::cg;

  if (libcgen_ == nullptr) {
    // throw std::runtime_error(
    //     "Error in LegacyCTarget::compile(): no library source generator is "
    //     "present. The code first needs to be generated.");
    generate_code();
  }
  
  if (cpu_compiler_ == nullptr) {
#if AUTOGEN_SYSTEM_WIN
    set_compiler_msvc();
#else
    set_compiler_clang();
#endif
  }
  cpu_compiler_->setSourcesFolder(sources_folder_);
  cpu_compiler_->setTemporaryFolder(temp_folder_);
  cpu_compiler_->setSaveToDiskFirst(true);
  if (debug_mode_) {
    cpu_compiler_->addCompileFlag("-g");
    cpu_compiler_->addCompileFlag("-O0");
  } else {
    cpu_compiler_->addCompileFlag("-O" + std::to_string(optimization_level_));
  }

  DynamicModelLibraryProcessor<BaseScalar> p(*libcgen_);
  p.setLibraryName(name() + "_" + std::to_string(type_));
  bool load_library = false;  // we do this in another step
  p.createDynamicLibrary(*cpu_compiler_, load_library);
  library_name_ = "./" + name() + "_" + std::to_string(type_);

  return true;
}
}  // namespace autogen