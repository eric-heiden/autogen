enable_testing()

include(GoogleTest)

add_executable(test_basic_codegen test_basic_codegen.cpp)
target_link_libraries(test_basic_codegen PRIVATE autogen gtest gtest_main)
add_test(AutogenTests test_basic_codegen)