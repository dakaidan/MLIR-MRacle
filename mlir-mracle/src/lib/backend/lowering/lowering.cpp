#include "mlir-mracle/backend/lowering/lowering.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir_mracle {

namespace {

// manual fixup for the OpenMP lowering since the conversion refuses to handle memref
void promoteOMPAtomicOperands(mlir::ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    module.walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::omp::AtomicWriteOp, mlir::omp::AtomicReadOp,
                       mlir::omp::AtomicCompareOp>(op))
            return;

        bool changed = false;
        llvm::SmallVector<mlir::Value> newOperands;
        newOperands.reserve(op->getNumOperands());
        for (mlir::Value operand : op->getOperands()) {
            mlir::Value replacement = operand;
            if (mlir::isa<mlir::MemRefType>(operand.getType())) {
                if (auto cast =
                        operand.getDefiningOp<mlir::UnrealizedConversionCastOp>();
                    cast && cast.getInputs().size() == 1) {
                    mlir::Value src = cast.getInputs().front();
                    if (mlir::isa<mlir::LLVM::LLVMPointerType>(src.getType())) {
                        replacement = src;
                    } else if (auto structTy =
                                   mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
                                       src.getType())) {
                        builder.setInsertionPoint(op);
                        replacement = mlir::LLVM::ExtractValueOp::create(
                            builder, op->getLoc(), src,
                            llvm::ArrayRef<int64_t>{1});
                    }
                }
            }
            changed |= (replacement != operand);
            newOperands.push_back(replacement);
        }
        if (changed) {
            op->setOperands(newOperands);
            for (mlir::Value operand : newOperands)
                if (auto cast =
                        operand.getDefiningOp<mlir::UnrealizedConversionCastOp>();
                    cast && cast->use_empty())
                    cast->erase();
        }
    });
}

} // namespace

mlir::LogicalResult lowerToLLVM(mlir::ModuleOp module, mlir::MLIRContext *ctx) {

    mlir::PassManager pm1(ctx);
    pm1.addPass(mlir::createLowerAffinePass()); // affine -> loops/memref

    if (mlir::failed(pm1.run(module)))
        return mlir::failure();

    mlir::PassManager pm2(ctx);
    pm2.addPass(mlir::createSCFToControlFlowPass()); // scf -> cf

    pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass()); // clean up unrealized casts
    if (mlir::failed(pm2.run(module)))
        return mlir::failure();

    // Custom pass to promote atomic operands to LLVM pointers.
    promoteOMPAtomicOperands(module);

    // convert to LLVM dialect
    mlir::PassManager pm3(ctx);
    pm3.addPass(mlir::createArithToLLVMConversionPass());       // arith -> llvm
    pm3.addPass(mlir::createConvertControlFlowToLLVMPass());    // cf -> llvm
    pm3.addPass(mlir::createConvertFuncToLLVMPass());           // func -> llvm
    pm3.addPass(mlir::createConvertOpenMPToLLVMPass());         // omp -> llvm
    pm3.addPass(mlir::createReconcileUnrealizedCastsPass());
    return pm3.run(module);
}

} // namespace mlir_mracle
