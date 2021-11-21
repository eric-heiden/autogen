#include "autogen/cg/target/legacy_c_target.h"

#include "autogen/utils/system.h"

namespace autogen {
void LegacyCTarget::forward(const std::vector<BaseScalar> &input,
                            std::vector<BaseScalar> &output) {
  assert(!library_name_.empty());
  main_model_->ForwardZero(input, output);
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
      main_model_->ForwardZero(local_inputs[i], outputs[i]);
    } else {
      static thread_local std::vector<BaseScalar> input;
      input = global_input;
      input.resize(global_input.size() + local_inputs[0].size());
      for (size_t j = 0; j < local_inputs[i].size(); ++j) {
        input[j + global_input.size()] = local_inputs[i][j];
      }
      main_model_->ForwardZero(input, outputs[i]);
    }
  }
}

void LegacyCTarget::jacobian(const std::vector<BaseScalar> &input,
                             std::vector<BaseScalar> &output) {
  main_model_->Jacobian(input, output);
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
      // main_model_->ForwardZero(local_inputs[i], outputs[i]);
      main_model_->Jacobian(local_inputs[i], outputs[i]);
    } else {
      static thread_local std::vector<BaseScalar> input;
      if (input.empty()) {
        input.resize(global_input.size());
        input.insert(input.begin(), global_input.begin(), global_input.end());
      }
      for (size_t j = 0; j < local_inputs[i].size(); ++j) {
        input[j + global_input.size()] = local_inputs[i][j];
      }
      main_model_->Jacobian(input, outputs[i]);
    }
  }
}

bool LegacyCTarget::load_library(const std::string &filename) {
  if (!cpu_library_) {
    cpu_library_loading_mutex_.lock();
    cpu_library_ = std::make_shared<DynamicLib>(filename + library_ext_);
    std::set<std::string> model_names = cpu_library_->getModelNames();
    std::cout << "Successfully loaded CPU library " << filename + library_ext_
              << std::endl;
    for (auto &name : model_names) {
      std::cout << "  Found model " << name << std::endl;
    }
    // load and wire up atomic functions in this library
    const auto &order = *CodeGenData::invocation_order;
    const auto &hierarchy = CodeGenData::call_hierarchy;
    lib_models_[this->name()] =
        GenericModelPtr(cpu_library_->model(this->name()).release());
    if (!lib_models_[this->name()]) {
      throw std::runtime_error("Failed to load model from library " + filename +
                               library_ext_);
    }
    // atomic functions to be added
    typedef std::pair<std::string, std::string> ParentChild;
    std::set<ParentChild> remaining_atomics;
    for (const std::string &s :
         lib_models_[this->name()]->getAtomicFunctionNames()) {
      remaining_atomics.insert(std::make_pair(this->name(), s));
    }
    while (!remaining_atomics.empty()) {
      ParentChild member = *(remaining_atomics.begin());
      const std::string &parent = member.first;
      const std::string &atomic_name = member.second;
      remaining_atomics.erase(remaining_atomics.begin());
      if (lib_models_.find(atomic_name) == lib_models_.end()) {
        std::cout << "  Adding atomic function " << atomic_name << std::endl;
        lib_models_[atomic_name] =
            GenericModelPtr(cpu_library_->model(atomic_name).release());
        for (const std::string &s :
             lib_models_[atomic_name]->getAtomicFunctionNames()) {
          remaining_atomics.insert(std::make_pair(atomic_name, s));
        }
      }
      auto &atomic_model = lib_models_[atomic_name];
      lib_models_[parent]->addAtomicFunction(atomic_model->asAtomic());
    }

    std::cout << "Loaded compiled model \"" << this->name() << "\" from \""
              << filename << "\".\n";
    cpu_library_loading_mutex_.unlock();
  }
  main_model_ = lib_models_[this->name()];
  return true;
}

void LegacyCTarget::set_compiler_clang(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("clang");
  }
  compiler_ = std::make_shared<ClangCompiler>(compiler_path);
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
  }
}

void LegacyCTarget::set_compiler_gcc(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("gcc");
  }
  compiler_ = std::make_shared<GccCompiler>(compiler_path);
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
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
  compiler_ = std::make_shared<MsvcCompiler>(compiler_path, linker_path);
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
  }
}

bool LegacyCTarget::generate_code_() {
  using namespace CppAD;
  using namespace CppAD::cg;

  auto *main_source_gen =
      new ModelCSourceGen<BaseScalar>(*(this->main_trace().tape), this->name());
  main_source_gen->setCreateForwardZero(generate_forward_);
  main_source_gen->setCreateJacobian(generate_jacobian_);
  libcgen_ =
      std::make_shared<ModelLibraryCSourceGen<BaseScalar>>(*main_source_gen);

  // reverse order of invocation to first generate code for innermost
  // functions
  const auto &order = CodeGenData::invocation_order();
  models_.clear();
  models_.push_back(main_source_gen);
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    FunctionTrace &trace = CodeGenData::traces()[*it];
    // trace.tape->optimize();
    auto *source_gen = new ModelCSourceGen<BaseScalar>(*(trace.tape), *it);
    source_gen->setCreateForwardZero(generate_forward_);
    // source_gen->setCreateSparseJacobian(generate_jacobian_);
    // source_gen->setCreateJacobian(generate_jacobian_);
    source_gen->setCreateForwardOne(generate_jacobian_);
    source_gen->setCreateReverseOne(generate_jacobian_);
    models_.push_back(source_gen);
    // we need a stable reference
    libcgen_->addModel(*(models_.back()));
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

  if (compiler_ == nullptr) {
#if AUTOGEN_SYSTEM_WIN
    set_compiler_msvc();
#else
    set_compiler_clang();
#endif
  }
  compiler_->setSourcesFolder(sources_folder_);
  compiler_->setTemporaryFolder(temp_folder_);
  compiler_->setSaveToDiskFirst(true);

// TODO check type of compiler here, not operating system
#if AUTOGEN_SYSTEM_WIN
  if (debug_mode_) {
    compiler_->addCompileFlag("/DEBUG:FULL");
    compiler_->addCompileLibFlag("/DEBUG:FULL");
    compiler_->addCompileFlag("/Od");
  } else {
    compiler_->addCompileFlag("/O" + std::to_string(optimization_level_));
  }
#else
  if (debug_mode_) {
    compiler_->addCompileFlag("-O0");
    compiler_->addCompileFlag("-g");
  } else {
    compiler_->addCompileFlag("-O" + std::to_string(optimization_level_));
  }
#endif

  CppAD::cg::JobTimer *timer = new CppAD::cg::JobTimer();

  timer->startingJob("", JobTimer::DYNAMIC_MODEL_LIBRARY);

  std::map<std::string, std::string> source_files =
      libcgen_->getLibrarySources();
  // for (const auto &[filename, content] : sources_) {
  //   if (std::find(source_filenames_.begin(), source_filenames_.end(),
  //                 filename) != source_filenames_.end()) {
  //     source_files[filename] = content;
  //   }
  // }

  try {
    // const std::map<std::string, CppAD::cg::ModelCSourceGen<BaseScalar> *>
    //     &models = libcgen_->getModels();
    auto mt_type = CppAD::cg::MultiThreadingType::NONE;
    for (const auto* model : models_) {
      const std::map<std::string, std::string> &modelSources =
          model->getSources();

      timer->startingJob("", CppAD::cg::JobTimer::COMPILING_FOR_MODEL);
      compiler_->compileSources(modelSources, true, timer);
      timer->finishedJob();
    }

    // const std::map<std::string, std::string> &sources =
    //     this->getLibrarySources();
    compiler_->compileSources(source_files, true, timer);

    library_name_ =
        source_folder_prefix_ + name() + "_" + std::to_string(type_);
    std::string libname = library_name_ + library_ext_;

    compiler_->buildDynamic(libname, timer);

  } catch (...) {
    compiler_->cleanup();
    library_name_ = "";
    throw;
  }
  compiler_->cleanup();

  timer->finishedJob();

  // library_name_ = source_folder_prefix_ + name() + "_" +
  // std::to_string(type_);
  //   DynamicModelLibraryProcessor<BaseScalar> p(*libcgen_);
  //   p.setLibraryName(library_name_);
  //   bool load_library = false;  // we do this in another step
  //   p.createDynamicLibrary(*compiler_, load_library);

  return true;
}
}  // namespace autogen