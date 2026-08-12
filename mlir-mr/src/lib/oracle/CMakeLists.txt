add_library(mlir_mr_oracle STATIC oracle.cpp)
add_library(MLIRMR::mlir_mr_oracle ALIAS mlir_mr_oracle)

set_property(TARGET mlir_mr_oracle PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_oracle PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_oracle PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_oracle PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(Threads REQUIRED)

target_link_libraries(mlir_mr_oracle PUBLIC
    Threads::Threads
    mlir_mr_context
)
