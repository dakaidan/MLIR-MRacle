add_library(mlir_mr_analysis STATIC analysis.cpp)
add_library(MLIRMR::mlir_mr_analysis ALIAS mlir_mr_analysis)

set_property(TARGET mlir_mr_analysis PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_analysis PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_analysis PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_analysis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_analysis PUBLIC
    mlir_mr_outcome
    mlir_mr_context
)
