# Grab all your test files
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

add_executable(conquer_test ${TEST_SOURCES})

target_compile_options(conquer_test PRIVATE -fno-rtti)
set_target_properties(conquer_test PROPERTIES ENABLE_EXPORTS TRUE)

target_link_libraries(conquer_test PRIVATE
        conquer_lib
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(conquer_test)