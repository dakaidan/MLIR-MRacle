include "mlir/Pass/PassBase.td"

def MetamorphicPass : Pass<"MetamorphicPass", "::mlir::ModuleOp"> {

    let summary = "Implements Metamorphic Transformations for Concurrent Memory Operations";

    let description = [{
        This pass implements metamorphic transformations for concurrent memory operations in MLIR.
        It analyzes the memory access patterns and applies transformations that preserve the semantics of the program
        while potentially altering its execution behavior. This is useful for testing and verifying the correctness
        of compiler passes that deal with concurrent memory access.
    }];

    let constructor = "::mlir::createMetamorphicPass()";

    let options = [
      Option<"seed", "seed", "int", "42",
             "Seed for the RNG that picks which metamorphic relation to apply">,
      Option<"transform", "transform", "std::string", "",
             "Comma-separated list of transforms to try (empty = pick at random)">,
      Option<"maxApply", "apply", "int", "1",
             "Limit on the number of successful transformations applied to a single function">,
      Option<"model", "model", "std::string", "",
             "Memory model gating the applicable transforms: empty = generic "
             "transforms only; \"armv8\" adds the ARMv8-specific ones">
    ];

    let dependentDialects = [
        "mlir::memref::MemRefDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::omp::OpenMPDialect"
    ];
}
