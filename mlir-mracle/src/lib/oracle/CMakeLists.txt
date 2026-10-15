add_library(mlir_mracle_oracle STATIC
    oracle.cpp
    distribution.cpp
    distribution_oracle.cpp
    calibrated.cpp
)
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

# The distribution module and the oracle header are public API, so the test
# links the library and exercises the same decision rules the pipeline runs.
add_executable(mlir_mracle_oracle_test oracle_test.cpp)

set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mracle_oracle_test
    PRIVATE MLIRMRACLE::mlir_mracle_oracle MLIRMRACLE::mlir_mracle_core)

add_test(NAME oracle_test COMMAND mlir_mracle_oracle_test)

add_executable(mlir_mracle_calibrated_test calibrated_test.cpp)

set_property(TARGET mlir_mracle_calibrated_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_calibrated_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_calibrated_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mracle_calibrated_test
    PRIVATE MLIRMRACLE::mlir_mracle_oracle MLIRMRACLE::mlir_mracle_core)

add_test(NAME calibrated_test COMMAND mlir_mracle_calibrated_test)

# The per-outcome oracle of the default method includes its own translation unit
# so the static judges are reachable; the library's copy is deliberately not
# linked into this executable.
add_executable(mlir_mracle_poisson_oracle_test oracle_poisson_test.cpp)

set_property(TARGET mlir_mracle_poisson_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_poisson_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_poisson_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mracle_poisson_oracle_test
    PRIVATE MLIRMRACLE::mlir_mracle_core)

add_test(NAME poisson_oracle_test COMMAND mlir_mracle_poisson_oracle_test)
