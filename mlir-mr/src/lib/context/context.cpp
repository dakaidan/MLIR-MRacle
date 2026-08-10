#include "mlir-mr/context/context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir-mr/passes/MetamorphicMemoryModelPass.h"
#include "mlir/Pass/Pass.h"

namespace mlir_mr {

MLIRSetup::MLIRSetup(int seed, int runNumber, std::string transform,
                     int maxApply)
    : pm(&mlirContext, mlir::ModuleOp::getOperationName()) {
    runInfo.seed = seed;
    runInfo.runNumber = runNumber;
    pm.addPass(::mlir::createMetamorphicMemoryModelPass(seed, &runInfo, transform,
                                                       maxApply));
    mlir::registerBuiltinDialectTranslation(mlirContext);
    mlir::registerLLVMDialectTranslation(mlirContext);

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerOpenMPDialectTranslation(registry);
    mlirContext.appendDialectRegistry(registry);
}

} // namespace mlir_mr
