add_library(mlir_mracle_jit STATIC jit.cpp)
add_library(MLIRMRACLE::mlir_mracle_jit ALIAS mlir_mracle_jit)

set_property(TARGET mlir_mracle_jit PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_jit PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_jit PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_jit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)

target_link_libraries(mlir_mracle_jit PUBLIC
    mlir_mracle_context
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    LLVMOrcJIT
    LLVMJITLink
    LLVMOrcTargetProcess
    LLVMExecutionEngine
    LLVMInstrumentation
    LLVMTransformUtils
    LLVMPasses
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_link_libraries(mlir_mracle_jit PUBLIC LLVMAArch64CodeGen)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    target_link_libraries(mlir_mracle_jit PUBLIC LLVMX86CodeGen)
endif()
