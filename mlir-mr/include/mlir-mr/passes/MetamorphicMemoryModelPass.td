include "mlir/Pass/PassBase.td"

def MetamorphicMemoryModelPass : Pass<"MetamorphicMemoryModelPass", "::mlir::func::FuncOp"> {

    let summary = "Implements Metamorphic Transformations for Concurrent Memory Operations";

    let description = [{
        This pass implements metamorphic transformations for concurrent memory operations in MLIR. 
        It analyzes the memory access patterns and applies transformations that preserve the semantics of the program 
        while potentially altering its execution behavior. This is useful for testing and verifying the correctness 
        of compiler passes that deal with concurrent memory access.
    }];

    let constructor = "::mlir::createMetamorphicMemoryModelPass()";

    let dependentDialects = [
        "mlir::memref::MemRefDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];
}