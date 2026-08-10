#include "mlir-mr/context/context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir-mr/passes/MetamorphicMemoryModelPass.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_mr {

std::string RunInfo::toString() const {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "run: " << runNumber << "\n";
    os << "seed: " << seed << "\n";
    os << "file: " << file << "\n";
    os << "requested-transforms: ";
    if (requestedTransforms.empty())
        os << "all\n";
    else
        os << llvm::join(requestedTransforms, ",") << "\n";
    if (transformApplied)
        for (const auto &at : appliedTransforms)
            os << "applied-transformation: " << at.name
               << " in function '" << at.targetFunction << "'\n";
    else
        os << "applied-transformation: none\n";
    if (!error.empty())
        os << "error: " << error << "\n";
    if (!warn.empty())
        os << "warn: " << warn << "\n";
    return buf;
}

llvm::json::Object RunInfo::toJson(bool includeMLIR) const {
    llvm::json::Object obj;
    obj["run"] = runNumber;
    obj["seed"] = seed;
    obj["file"] = file;
    llvm::json::Array requested;
    if (requestedTransforms.empty())
        requested.push_back("all");
    else
        for (const auto &r : requestedTransforms)
            requested.push_back(r);
    obj["requested_transforms"] = std::move(requested);
    llvm::json::Array applied;
    for (const auto &at : appliedTransforms) {
        llvm::json::Object entry;
        entry["name"] = at.name;
        entry["target_function"] = at.targetFunction;
        applied.push_back(std::move(entry));
    }
    obj["applied_transforms"] = std::move(applied);
    obj["transform_applied"] = transformApplied;
    if (!error.empty())
        obj["error"] = error;
    if (!warn.empty())
        obj["warn"] = warn;
    if (includeMLIR)
        obj["mlir_output"] = mlirOutput;
    return obj;
}

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
