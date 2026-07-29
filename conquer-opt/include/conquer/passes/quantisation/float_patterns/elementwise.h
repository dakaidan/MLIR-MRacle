#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::float_quant {
// Handles: Add, Sub, Mul, ArithmeticRightShift, Pow, Maximum, Minimum

template <typename OpTy> struct QuantiseElementwisePattern : mlir::RewritePattern {
    QuantiseElementwisePattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();

        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        L_TRACE("Applying QuantiseElementwisePattern to: " << compact(op));

        const auto layer_target = get_target_type(*rewriter.getContext(), op, quantPolicy);
        if (!layer_target.target_weight_type.has_value() && !layer_target.target_activation_type.has_value()) return mlir::failure();

        // Nullopt fallback to Float32Type
        mlir::Type target_wgt_type = layer_target.target_weight_type.value_or(mlir::Float32Type::get(rewriter.getContext()));
        mlir::Type target_act_type = layer_target.target_activation_type.value_or(mlir::Float32Type::get(rewriter.getContext()));

        const bool weight_is_int = target_wgt_type.isIntOrIndex();
        const bool activation_is_int = target_act_type.isIntOrIndex();

        if (weight_is_int && activation_is_int) return mlir::failure();

        mlir::Type compute_elem_type = get_highest_precision_type(layer_target);
        if (compute_elem_type.isIntOrIndex()) {
            // Mixed mode fallback: force it to the available float target
            compute_elem_type = !weight_is_int ? target_wgt_type : target_act_type;
        }

        const auto plan = planRequestedFloatMode<OpTy>(op, compute_elem_type, get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

        const mlir::Value lhs = op->getOperand(0);
        const mlir::Value rhs = op->getOperand(1);

        const std::optional lhsStorage = is_weight(lhs) ? target_wgt_type : target_act_type;
        const std::optional rhsStorage = is_weight(rhs) ? target_wgt_type : target_act_type;

        const mlir::Value quantisedLhs = applyMixedPrecisionCast(lhs, lhsStorage, plan->operandExecutionType, rewriter, op->getLoc());
        const mlir::Value quantisedRhs = applyMixedPrecisionCast(rhs, rhsStorage, plan->operandExecutionType, rewriter, op->getLoc());

        llvm::SmallVector<mlir::Value, 4> newOperands = {quantisedLhs, quantisedRhs};

        // Safely carry over operands like 'shift' for tosa.mul without modifying their types
        for (unsigned i = 2; i < op->getNumOperands(); ++i) {
            newOperands.push_back(op->getOperand(i));
        }

        const auto newReturnType = mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        auto newOp = OpTy::create(rewriter, op->getLoc(), newReturnType, newOperands, op->getAttrs());
        newOp->setAttrs(op->getAttrs());
        newOp->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut = ensureType(newOp->getResult(0), currentResultType.getElementType(), rewriter, op->getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

inline void populateElementWisePatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseElementwisePattern<mlir::tosa::AddOp>>(context, policy);
    patterns.add<QuantiseElementwisePattern<mlir::tosa::SubOp>>(context, policy);
    patterns.add<QuantiseElementwisePattern<mlir::tosa::MulOp>>(context, policy);
    patterns.add<QuantiseElementwisePattern<mlir::tosa::MaximumOp>>(context, policy);
    patterns.add<QuantiseElementwisePattern<mlir::tosa::MinimumOp>>(context, policy);
    patterns.add<QuantiseElementwisePattern<mlir::tosa::PowOp>>(context, policy);
}
} // namespace conquer