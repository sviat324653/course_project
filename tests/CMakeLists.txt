add_executable(lib_tests
  tests.cpp
)

target_link_libraries(lib_tests PRIVATE component_lib gtest_main)
include(GoogleTest)

gtest_discover_tests(lib_tests PROPERTIES ENVIRONMENT "GTEST_COLOR=yes")
