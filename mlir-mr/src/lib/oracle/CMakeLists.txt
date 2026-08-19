add_library(mlir_mr_oracle STATIC oracle.cpp new_oracle.cpp)
add_library(MLIRMR::mlir_mr_oracle ALIAS mlir_mr_oracle)

set_property(TARGET mlir_mr_oracle PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_oracle PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_oracle PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_oracle PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(mlir_mr_oracle PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    mlir_mr_context
)

add_executable(mlir_mr_oracle_test oracle_test.cpp)

set_property(TARGET mlir_mr_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mr_oracle_test PRIVATE MLIRMR::mlir_mr_oracle)

# includes new_oracle.cpp (see the #include in the test) so the static judges
# are testable; the library is deliberately not linked (it would duplicate
# newOracleCompare)
add_executable(mlir_mr_new_oracle_test
    new_oracle_test.cpp
)

set_property(TARGET mlir_mr_new_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_new_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_new_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mr_new_oracle_test PRIVATE MLIRMR::mlir_mr_context)
