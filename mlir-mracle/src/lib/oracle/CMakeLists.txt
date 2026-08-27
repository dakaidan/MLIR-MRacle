add_library(mlir_mracle_oracle STATIC oracle.cpp)
add_library(MLIRMRACLE::mlir_mracle_oracle ALIAS mlir_mracle_oracle)

set_property(TARGET mlir_mracle_oracle PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_oracle PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_oracle PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_oracle PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(mlir_mracle_oracle PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    mlir_mracle_core
)

# The test includes oracle.cpp (see the #include in the test) so the static
# judges are testable; the library is deliberately not linked (it would
# duplicate oracleCompare)
add_executable(mlir_mracle_oracle_test oracle_test.cpp)

set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mracle_oracle_test PRIVATE MLIRMRACLE::mlir_mracle_core)

add_test(NAME oracle_test COMMAND mlir_mracle_oracle_test)
