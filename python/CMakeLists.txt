add_subdirectory(pybind11)
pybind11_add_module(autogen_python main.cpp)
#pybind11_add_module(autogen_python src/main.cpp ${AUTOGEN_HDRS})

#target_include_directories(autogen_python INTERFACE ../include)
target_link_libraries(autogen_python PRIVATE autogen)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(autogen_python PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
