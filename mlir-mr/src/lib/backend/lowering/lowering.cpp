#include "mlir-mr/backend/lowering/lowering.h"

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

namespace mlir_mr {

namespace {

/// manual fixup for the OpenMP lowering since the conversion refuses to handle memref
void promoteOMPAtomicOperands(mlir::ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    module.walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::omp::AtomicWriteOp, mlir::omp::AtomicReadOp>(op))
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
    // memref must be lowered before OpenMP: omp.atomic.write/omp.flush carry
    // memref operands that the OpenMP conversion refuses to handle, and the
    // LLVMIR translation needs LLVM pointers for them.
    mlir::PassManager pm(ctx);
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    if (mlir::failed(pm.run(module)))
        return mlir::failure();
    promoteOMPAtomicOperands(module);
    mlir::PassManager pm2(ctx);
    pm2.addPass(mlir::createArithToLLVMConversionPass());
    pm2.addPass(mlir::createConvertOpenMPToLLVMPass());
    pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm2.addPass(mlir::createConvertFuncToLLVMPass());
    pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
    return pm2.run(module);
}

} // namespace mlir_mr
