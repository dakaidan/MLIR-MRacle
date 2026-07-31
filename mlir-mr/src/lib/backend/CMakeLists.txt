add_library(mlir_mr_backend STATIC executor.cpp)
add_library(MLIRMR::mlir_mr_backend ALIAS mlir_mr_backend)

set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_backend PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mlir_mr_backend PUBLIC
    LLVMCore
    LLVMSupport
    LLVMOrcJIT
    LLVMJITLink
    LLVMOrcTargetProcess
    LLVMExecutionEngine
)
