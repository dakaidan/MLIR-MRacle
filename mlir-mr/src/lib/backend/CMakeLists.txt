add_library(mlir_mr_backend STATIC jit.cpp)
add_library(MLIRMR::mlir_mr_backend ALIAS mlir_mr_backend)

set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_backend PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_backend PUBLIC
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    LLVMOrcJIT
    LLVMJITLink
    LLVMOrcTargetProcess
    LLVMExecutionEngine
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_link_libraries(mlir_mr_backend PUBLIC LLVMAArch64CodeGen)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    target_link_libraries(mlir_mr_backend PUBLIC LLVMX86CodeGen)
endif()