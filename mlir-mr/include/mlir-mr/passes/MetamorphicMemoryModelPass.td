include "mlir/Pass/PassBase.td"

def MetamorphicMemoryModelPass : Pass<"MetamorphicMemoryModelPass", "::mlir::ModuleOp"> {

    let summary = "Implements Metamorphic Transformations for Concurrent Memory Operations";

    let description = [{
        This pass implements metamorphic transformations for concurrent memory operations in MLIR.
        It analyzes the memory access patterns and applies transformations that preserve the semantics of the program
        while potentially altering its execution behavior. This is useful for testing and verifying the correctness
        of compiler passes that deal with concurrent memory access.
    }];

    let constructor = "::mlir::createMetamorphicMemoryModelPass()";

    let options = [
      Option<"seed", "seed", "int", "42",
             "Seed for the RNG that picks which metamorphic relation to apply">,
      ListOption<"transforms", "transforms", "std::string",
             "Comma-separated list of transforms to select from: "
             "multiset-permutation, load-reordering, insert-fence. "
             "Empty (default) uses all three.">,
      Option<"debug", "debug", "bool", "false",
             "Print which transformation was applied and in which function">
    ];

    let dependentDialects = [
        "mlir::memref::MemRefDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::omp::OpenMPDialect"
    ];
}
