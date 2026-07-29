#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::float_quant {

// Handles: ArgMax, Equal, Greater, GreaterEqual

struct QuantiseArgMaxPattern : mlir::OpRewritePattern<mlir::tosa::ArgMaxOp> {
    QuantiseArgMaxPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::ArgMaxOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised"))
            return mlir::failure();

        if (!should_quantise(op, quantPolicy))
            return mlir::failure();

        L_TRACE("Applying QuantiseArgMaxPattern to: " << compact(op));

        const auto [_, target_activation_type] = get_target_type(*rewriter.getContext(), op, quantPolicy);

        mlir::Type target_act_type =
            target_activation_type.value_or(mlir::Float32Type::get(rewriter.getContext()));

        if (target_act_type.isIntOrIndex())
            return mlir::failure();

        const auto plan = planTosaFloatMode(op, target_act_type,get_should_squash_acc(op, quantPolicy));
        if (!plan)
            return mlir::failure();

        const auto currentResultType =
            llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
        if (!currentResultType)
            return mlir::failure();

        // TOSA ArgMax result should already be i32.
        if (currentResultType.getElementType() != plan->resultType)
            return mlir::failure();

        const mlir::Value quantisedInput =
            applyMixedPrecisionCast(op.getInput(), target_act_type, plan->operandExecutionType,
                                    rewriter, op.getLoc());

        const auto newReturnType =
            mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        mlir::OperationState state(op.getLoc(), op->getName().getStringRef());
        state.addOperands({quantisedInput});
        state.addTypes({newReturnType});
        state.addAttributes(op->getAttrs());

        mlir::Operation *newOp = rewriter.create(state);
        newOp->setAttrs(op->getAttrs());
        newOp->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        rewriter.replaceOp(op, newOp->getResult(0));
        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

template <typename OpTy>
struct QuantiseComparisonPattern : mlir::RewritePattern {
    QuantiseComparisonPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised"))
            return mlir::failure();

        if (!should_quantise(op, quantPolicy))
            return mlir::failure();

        const auto layer_target = get_target_type(*rewriter.getContext(), op, quantPolicy);
        if (!layer_target.target_weight_type.has_value() &&
            !layer_target.target_activation_type.has_value()) {
            return mlir::failure();
        }

        mlir::Type target_wgt_type =
            layer_target.target_weight_type.value_or(mlir::Float32Type::get(rewriter.getContext()));
        mlir::Type target_act_type =
            layer_target.target_activation_type.value_or(mlir::Float32Type::get(rewriter.getContext()));

        const bool weight_is_int = target_wgt_type.isIntOrIndex();
        const bool activation_is_int = target_act_type.isIntOrIndex();

        if (weight_is_int && activation_is_int)
            return mlir::failure();

        mlir::Type compute_elem_type = get_highest_precision_type(layer_target);
        if (!compute_elem_type || compute_elem_type.isIntOrIndex()) {
            compute_elem_type = !weight_is_int ? target_wgt_type : target_act_type;
        }

        const auto plan = planTosaFloatMode(op, compute_elem_type, get_should_squash_acc(op, quantPolicy));
        if (!plan)
            return mlir::failure();

        const auto currentResultType =
            llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!currentResultType)
            return mlir::failure();

        // Comparison results should already be bool/i1.
        if (currentResultType.getElementType() != plan->resultType)
            return mlir::failure();

        const mlir::Value lhs = op->getOperand(0);
        const mlir::Value rhs = op->getOperand(1);

        const std::optional lhsStorage = is_weight(lhs) ? target_wgt_type : target_act_type;
        const std::optional rhsStorage = is_weight(rhs) ? target_wgt_type : target_act_type;

        const mlir::Value quantisedLhs =
            applyMixedPrecisionCast(lhs, lhsStorage, plan->operandExecutionType,
                                    rewriter, op->getLoc());
        const mlir::Value quantisedRhs =
            applyMixedPrecisionCast(rhs, rhsStorage, plan->operandExecutionType,
                                    rewriter, op->getLoc());

        const auto newReturnType =
            mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
        state.addOperands({quantisedLhs, quantisedRhs});
        state.addTypes({newReturnType});
        state.addAttributes(op->getAttrs());

        mlir::Operation *newOp = rewriter.create(state);
        newOp->setAttrs(op->getAttrs());
        newOp->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        rewriter.replaceOp(op, newOp->getResult(0));
        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

inline void populateIndexPatterns(mlir::RewritePatternSet &patterns,
                                  const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseArgMaxPattern>(context, policy);

    patterns.add<QuantiseComparisonPattern<mlir::tosa::EqualOp>>(context, policy);
    patterns.add<QuantiseComparisonPattern<mlir::tosa::GreaterOp>>(context, policy);
    patterns.add<QuantiseComparisonPattern<mlir::tosa::GreaterEqualOp>>(context, policy);
}

} // namespace conquer
