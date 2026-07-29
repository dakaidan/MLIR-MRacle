#include "conquer/passes/analysis/activation_instrumentation_pass.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

void conquer::ActivationInstrumentationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    auto i32Type = builder.getI32Type();
    auto unrankedTensorType = mlir::UnrankedTensorType::get(builder.getF32Type());
    const auto funcType = builder.getFunctionType({i32Type, unrankedTensorType}, {});

    builder.setInsertionPointToStart(module.getBody());
    auto externalFunc = mlir::func::FuncOp::create(
        builder, module.getLoc(), "conquer_record_activation_stats", funcType);

    externalFunc.setPrivate();
    externalFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    externalFunc.setArgAttr(1, "bufferization.access", builder.getStringAttr("read"));

    int32_t layerId = 0;

    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp == externalFunc) return;

        std::vector<mlir::Value> valuesToInstrument;

        // 1. Instrument Inputs
        if (!funcOp.getBody().empty()) {
            for (const auto arg : funcOp.getArguments()) {
                if (llvm::isa<mlir::TensorType>(arg.getType())) {
                    valuesToInstrument.push_back(arg);
                }
            }
        }

        // 2. Instrument Operations
        funcOp.walk([&](mlir::Operation *op) {
            if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return;

            // THE FOOLPROOF FIX: Automatically skip all constants natively
            if (op->hasTrait<mlir::OpTrait::ConstantLike>()) return;

            if (op->getNumResults() == 0) return;

            if (llvm::isa<mlir::TensorType>(op->getResult(0).getType())) {
                valuesToInstrument.push_back(op->getResult(0));
            }
        });

        // 3. Apply the call
        for (mlir::Value val : valuesToInstrument) {
            if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
                builder.setInsertionPointToStart(arg.getOwner());
            } else {
                builder.setInsertionPointAfter(val.getDefiningOp());
            }

            auto loc = val.getLoc();
            mlir::Value currentVal = val;
            auto tensorType = llvm::cast<mlir::TensorType>(currentVal.getType());

            if (!tensorType.getElementType().isF32()) {
                auto f32Type = tensorType.clone(builder.getF32Type());
                currentVal = mlir::tosa::CastOp::create(builder, loc, f32Type, currentVal);
            }

            auto idConst = mlir::arith::ConstantIntOp::create(builder, loc, layerId++, 32);
            auto castedTensor = mlir::tensor::CastOp::create(builder, loc, unrankedTensorType, currentVal);
            mlir::func::CallOp::create(builder, loc, externalFunc, mlir::ValueRange{idConst, castedTensor});
        }
    });
}