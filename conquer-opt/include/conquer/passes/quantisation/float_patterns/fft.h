#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::float_quant {

template <typename OpTy> struct QuantiseFFTPattern : mlir::RewritePattern {
    QuantiseFFTPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        L_TRACE("Applying QuantiseFFTPattern to: " << compact(op));

        const auto [_, target_activation_type] = get_target_type(*rewriter.getContext(), op, quantPolicy);
        if (!target_activation_type.has_value()) return mlir::failure();

        if (target_activation_type.value().isIntOrIndex()) return mlir::failure();

        const auto plan = planTosaFloatMode(op, target_activation_type.value(), get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        // 1 input for RFFT, 2 inputs for FFT)
        llvm::SmallVector<mlir::Value, 2> newOperands;
        for (const auto operand : op->getOperands()) {
            newOperands.push_back(applyMixedPrecisionCast(operand, target_activation_type, plan->operandExecutionType, rewriter, op->getLoc()));
        }

        llvm::SmallVector<mlir::Type, 2> newResultTypes;
        for (auto result : op->getResults()) {
            auto rankedTy = llvm::cast<mlir::RankedTensorType>(result.getType());
            newResultTypes.push_back(mlir::RankedTensorType::get(rankedTy.getShape(), plan->resultType));
        }

        mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultTypes);
        state.addAttributes(op->getAttrs());

        const auto newOp = rewriter.create(state);
        newOp->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        llvm::SmallVector<mlir::Value, 2> finalResults;
        for (auto [oldRes, newRes] : llvm::zip(op->getResults(), newOp->getResults())) {
            const auto origElemType = llvm::cast<mlir::RankedTensorType>(oldRes.getType()).getElementType();
            finalResults.push_back(ensureType(newRes, origElemType, rewriter, op->getLoc()));
        }

        rewriter.replaceOp(op, finalResults);
        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

inline void populateFFTPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    patterns.add<QuantiseFFTPattern<mlir::tosa::FFT2dOp>>(patterns.getContext(), policy);
    patterns.add<QuantiseFFTPattern<mlir::tosa::RFFT2dOp>>(patterns.getContext(), policy);
}

}