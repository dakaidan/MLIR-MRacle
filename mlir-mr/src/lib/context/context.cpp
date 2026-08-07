#include "mlir-mr/context/context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir-mr/passes/MetamorphicMemoryModelPass.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_mr {

std::string RunInfo::toString() const {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "run: " << runNumber << "\n";
    os << "seed: " << seed << "\n";
    os << "file: " << file << "\n";
    os << "requested-transforms: "
       << (requestedTransforms.empty() ? "all" : requestedTransforms) << "\n";
    if (transformApplied)
        os << "applied-transformation: " << appliedTransform
           << " in function '" << targetFunction << "'\n";
    else
        os << "applied-transformation: none\n";
    if (!error.empty())
        os << "error: " << error << "\n";
    return buf;
}

llvm::json::Object RunInfo::toJson(bool includeMLIR) const {
    llvm::json::Object obj;
    obj["run"] = runNumber;
    obj["seed"] = seed;
    obj["file"] = file;
    obj["requested_transforms"] =
        requestedTransforms.empty() ? "all" : requestedTransforms;
    if (transformApplied) {
        obj["applied_transform"] = appliedTransform;
        obj["target_function"] = targetFunction;
        obj["transform_applied"] = true;
    } else {
        obj["applied_transform"] = "none";
        obj["transform_applied"] = false;
    }
    if (!error.empty())
        obj["error"] = error;
    if (includeMLIR)
        obj["mlir_output"] = mlirOutput;
    return obj;
}

MLIRSetup::MLIRSetup(int seed, int runNumber, std::string transform)
    : pm(&mlirContext, mlir::ModuleOp::getOperationName()) {
    runInfo.seed = seed;
    runInfo.runNumber = runNumber;
    pm.addPass(::mlir::createMetamorphicMemoryModelPass(seed, &runInfo, transform));
    mlir::registerBuiltinDialectTranslation(mlirContext);
    mlir::registerLLVMDialectTranslation(mlirContext);

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerOpenMPDialectTranslation(registry);
    mlirContext.appendDialectRegistry(registry);
}

} // namespace mlir_mr
