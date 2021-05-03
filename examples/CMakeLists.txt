add_executable(basic_codegen basic_codegen.cpp)
target_link_libraries(basic_codegen autogen)

add_executable(loop_codegen loop_codegen.cpp)
target_link_libraries(loop_codegen autogen)

add_executable(test_loop test_loop.cpp)
target_link_libraries(test_loop autogen)

add_executable(regex_testing regex_testing.cpp)