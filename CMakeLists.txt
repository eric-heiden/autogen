project(autogen)
cmake_minimum_required(VERSION 3.13)

cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

include(ExternalProject)

# Add CppAD as an external project and an imported target.  
ExternalProject_Add(CppAD-Project
PREFIX CppAD
SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/CppAD
CMAKE_ARGS
  -Dcppad_prefix=${CMAKE_CURRENT_BINARY_DIR}/CppAD
)
add_library(CppAD INTERFACE IMPORTED)
add_dependencies(CppAD CppAD-Project)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CppAD/include)
target_include_directories(CppAD INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/CppAD/include)
target_compile_definitions(CppAD INTERFACE -DUSE_CPPAD)

# Same for CppADCodeGen
ExternalProject_Add(CppADCodeGen-Project
PREFIX CppADCodeGen
SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/CppADCodeGen
CMAKE_ARGS
  -DCPPAD_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/CppAD/include
  -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/CppADCodeGen
  -DGOOGLETEST_GIT=ON
)
add_dependencies(CppADCodeGen-Project CppAD)
add_library(CppADCodeGen INTERFACE IMPORTED)
add_dependencies(CppADCodeGen CppAD-Project CppADCodeGen-Project)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CppAD/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CppADCodeGen/include)
target_include_directories(CppADCodeGen
INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/CppAD/include
  ${CMAKE_CURRENT_BINARY_DIR}/CppADCodeGen/include
)

add_library(autogen INTERFACE)
target_include_directories(autogen INTERFACE include)
target_link_libraries(autogen INTERFACE CppADCodeGen Eigen3::Eigen)

add_subdirectory(examples)