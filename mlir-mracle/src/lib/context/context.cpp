#include "mlir-mracle/context/context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir-mracle/passes/MetamorphicPass.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir_mracle {

std::string outcomeRelationToString(OutcomeRelation relation) {
    switch (relation) {
    case OutcomeRelation::Equality:
        return "equality";
    case OutcomeRelation::Subset:
        return "subset";
    case OutcomeRelation::Superset:
        return "superset";
    }
    llvm_unreachable("unknown outcome relation");
}

bool outcomeRelationFromString(const std::string &s,
                               OutcomeRelation &relation) {
    if (s == "equality") {
        relation = OutcomeRelation::Equality;
        return true;
    }
    if (s == "subset") {
        relation = OutcomeRelation::Subset;
        return true;
    }
    if (s == "superset") {
        relation = OutcomeRelation::Superset;
        return true;
    }
    return false;
}

void initializeMLIRContext(mlir::MLIRContext &ctx) {
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerOpenMPDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
}

MLIRSetup::MLIRSetup(int seed, int runNumber, std::string transform,
                     int maxApply, std::string model)
    : pm(&mlirContext, mlir::ModuleOp::getOperationName()) {
    runInfo.seed = seed;
    runInfo.runNumber = runNumber;
    pm.addPass(::mlir::createMetamorphicPass(seed, &runInfo, transform,
                                            maxApply, model));
    initializeMLIRContext(mlirContext);
}

} // namespace mlir_mracle
