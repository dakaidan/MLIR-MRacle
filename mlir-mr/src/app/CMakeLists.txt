add_executable(mlir_mr_opt main.cpp)
target_link_libraries(mlir_mr_opt PRIVATE MLIRMR::mlir_mr)