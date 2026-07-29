#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::float_quant {
// Handles: avg_pool2d, reduce_sum, reduce_product, reduce_max, reduce_min

struct QuantiseAvgPool2dPattern : mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp> {
    QuantiseAvgPool2dPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::AvgPool2dOp op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        L_TRACE("Applying QuantiseAvgPool2dPattern to: " << compact(op));

        auto [_, targetActivationType] = get_target_type(*rewriter.getContext(), op, quantPolicy);
        if (!targetActivationType.has_value()) return mlir::failure();

        if (targetActivationType.value().isIntOrIndex()) return mlir::failure(); // Leave for Int pass

        const auto plan = planTosaFloatMode(op, targetActivationType.value(), get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
        if (!currentResultType) return mlir::failure();

        // Apply casts.
        mlir::Value quantisedInput = applyMixedPrecisionCast(
            op.getInput(), targetActivationType, plan->operandExecutionType, rewriter, op.getLoc()
        );

        // Zero points must match execution type exactly in float TOSA.
        mlir::Value inputZp = ensureType(op.getInputZp(), plan->operandExecutionType, rewriter, op.getLoc());
        mlir::Value outputZp = ensureType(op.getOutputZp(), plan->operandExecutionType, rewriter, op.getLoc());

        const auto newReturnType = mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        mlir::OperationState state(op.getLoc(), op->getName().getStringRef());
        state.addOperands({quantisedInput, inputZp, outputZp});
        state.addTypes(newReturnType);

        // Preserve all original attrs, but replace acc_type if the new plan asks for one.
        bool sawAccType = false;
        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "acc_type") {
                sawAccType = true;
                if (plan->accumulatorType) {
                    state.addAttribute(attr.getName(), mlir::TypeAttr::get(*plan->accumulatorType));
                } else {
                    state.addAttribute(attr.getName(), attr.getValue());
                }
            } else {
                state.addAttribute(attr.getName(), attr.getValue());
            }
        }

        if (!sawAccType && plan->accumulatorType) {
            state.addAttribute("acc_type", mlir::TypeAttr::get(*plan->accumulatorType));
        }

        mlir::Operation *newAvgPool = rewriter.create(state);
        newAvgPool->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut =
            ensureType(newAvgPool->getResult(0), currentResultType.getElementType(), rewriter, op.getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

// Handles: reduce_sum, reduce_product, reduce_max, reduce_min
template <typename OpTy>
struct QuantiseGenericReducePattern : mlir::RewritePattern {
    QuantiseGenericReducePattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        const auto [_, targetActivationType] = get_target_type(*rewriter.getContext(), op, quantPolicy);
        if (!targetActivationType.has_value()) return mlir::failure();

        if (targetActivationType.value().isIntOrIndex()) return mlir::failure();

        const auto plan = planTosaFloatMode(op, targetActivationType.value(), get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!currentResultType) return mlir::failure();

        // Reductions take a single input tensor (operand 0).
        mlir::Value quantisedInput = applyMixedPrecisionCast(
            op->getOperand(0), targetActivationType, plan->operandExecutionType, rewriter, op->getLoc()
        );

        const auto newReturnType = mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
        state.addOperands({quantisedInput});
        state.addTypes(newReturnType);
        state.addAttributes(op->getAttrs());

        mlir::Operation *newOp = rewriter.create(state);
        newOp->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut =
            ensureType(newOp->getResult(0), currentResultType.getElementType(), rewriter, op->getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

inline void populateReductionPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseAvgPool2dPattern>(context, policy);

    patterns.add<QuantiseGenericReducePattern<mlir::tosa::ReduceSumOp>>(context, policy);
    patterns.add<QuantiseGenericReducePattern<mlir::tosa::ReduceProductOp>>(context, policy);
    patterns.add<QuantiseGenericReducePattern<mlir::tosa::ReduceMaxOp>>(context, policy);
    patterns.add<QuantiseGenericReducePattern<mlir::tosa::ReduceMinOp>>(context, policy);
}
}