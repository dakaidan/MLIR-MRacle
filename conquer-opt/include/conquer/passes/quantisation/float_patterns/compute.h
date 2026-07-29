#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::float_quant {
// Handles: conv2d, conv3d, depthwise_conv2d, transpose_conv2d, matmul

template <typename OpTy>
struct QuantiseGenericConvPattern : mlir::RewritePattern {
    QuantiseGenericConvPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();

        auto typedOp = llvm::cast<OpTy>(op);
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        L_INFO("Quantising " << compact(op) << " to float.");

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

        // Leave integer passes for later
        if (weight_is_int && activation_is_int) return mlir::failure();

        mlir::Type compute_elem_type = get_highest_precision_type(layer_target);
        if (!compute_elem_type || compute_elem_type.isIntOrIndex()) {
            compute_elem_type = !weight_is_int ? target_wgt_type : target_act_type;
        }

        const auto plan = planTosaFloatMode(
            op, compute_elem_type, get_should_squash_acc(op, quantPolicy)
        );
        if (!plan) {
            L_DEBUG("  Failed to find float quantisation plan for " << compact(op));
            return mlir::failure();
        }

        L_DEBUG("  Using float execution type " << plan->operandExecutionType << " for " << compact(op));

        const auto currentResultType =
            llvm::dyn_cast<mlir::RankedTensorType>(typedOp.getResult().getType());
        if (!currentResultType) return mlir::failure();

        // Apply casts
        mlir::Value quantisedLhs = applyMixedPrecisionCast(
            typedOp.getInput(), target_act_type, plan->operandExecutionType, rewriter, op->getLoc()
        );
        mlir::Value quantisedRhs = applyMixedPrecisionCast(
            typedOp.getWeight(), target_wgt_type, plan->operandExecutionType, rewriter, op->getLoc()
        );

        const mlir::Type biasExecType = plan->biasType.value_or(plan->operandExecutionType);
        mlir::Value newBias = ensureType(typedOp.getBias(), biasExecType, rewriter, op->getLoc());

        const auto newReturnType =
            mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);
        const mlir::TypeAttr newAccType = mlir::TypeAttr::get(plan->accumulatorType.value());

        auto newConv = OpTy::create(
            rewriter, op->getLoc(), newReturnType, quantisedLhs, quantisedRhs, newBias,
            typedOp.getPadAttr(), typedOp.getStrideAttr(), typedOp.getDilationAttr(), newAccType
        );

        // Preserve all attrs, then restore the explicitly managed one.
        newConv->setAttrs(op->getAttrs());
        newConv->setAttr("acc_type", newAccType);
        newConv->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut =
            ensureType(newConv.getResult(), currentResultType.getElementType(), rewriter, op->getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

using QuantiseConv2DPattern = QuantiseGenericConvPattern<mlir::tosa::Conv2DOp>;
using QuantiseConv3DPattern = QuantiseGenericConvPattern<mlir::tosa::Conv3DOp>;
using QuantiseDepthwiseConv2DPattern = QuantiseGenericConvPattern<mlir::tosa::DepthwiseConv2DOp>;

struct QuantiseTransposeConv2DPattern : mlir::OpRewritePattern<mlir::tosa::TransposeConv2DOp> {
    QuantiseTransposeConv2DPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::TransposeConv2DOp op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

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
        if (weight_is_int && activation_is_int) return mlir::failure();

        mlir::Type compute_elem_type = get_highest_precision_type(layer_target);
        if (!compute_elem_type || compute_elem_type.isIntOrIndex()) {
            compute_elem_type = !weight_is_int ? target_wgt_type : target_act_type;
        }

        const auto plan = planTosaFloatMode(
            op, compute_elem_type, get_should_squash_acc(op, quantPolicy)
        );
        if (!plan) {
            L_DEBUG("  Failed to find float quantisation plan for " << compact(op));
            return mlir::failure();
        }

        L_DEBUG("  Using float execution type " << plan->operandExecutionType << " for " << compact(op));

        const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
        if (!currentResultType) return mlir::failure();

        const mlir::Value quantisedLhs = applyMixedPrecisionCast(
            op.getInput(), target_act_type, plan->operandExecutionType, rewriter, op.getLoc()
        );
        const mlir::Value quantisedRhs = applyMixedPrecisionCast(
            op.getWeight(), target_wgt_type, plan->operandExecutionType, rewriter, op.getLoc()
        );

        const mlir::Type biasExecType = plan->biasType.value_or(plan->operandExecutionType);
        const mlir::Value newBias = ensureType(op.getBias(), biasExecType, rewriter, op.getLoc());

        const auto newReturnType =
            mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);
        const mlir::TypeAttr newAccType = mlir::TypeAttr::get(plan->accumulatorType.value());

        auto newConv = mlir::tosa::TransposeConv2DOp::create(
            rewriter, op.getLoc(), newReturnType, quantisedLhs, quantisedRhs,
            newBias, op.getOutPadAttr(), op.getStrideAttr(), newAccType
        );

        // Preserve all attrs, then restore the explicitly managed one.
        newConv->setAttrs(op->getAttrs());
        newConv->setAttr("acc_type", newAccType);
        newConv->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut =
            ensureType(newConv.getResult(), currentResultType.getElementType(), rewriter, op.getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

struct QuantiseMatMulPattern : mlir::OpRewritePattern<mlir::tosa::MatMulOp> {
    QuantiseMatMulPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::MatMulOp op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.float.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

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
        if (weight_is_int && activation_is_int) return mlir::failure();

        mlir::Type compute_elem_type = get_highest_precision_type(layer_target);
        if (!compute_elem_type || compute_elem_type.isIntOrIndex()) {
            compute_elem_type = !weight_is_int ? target_wgt_type : target_act_type;
        }

        const auto plan = planTosaFloatMode(
            op, compute_elem_type, get_should_squash_acc(op, quantPolicy)
        );
        if (!plan) {
            L_DEBUG("  Failed to find float quantisation plan for " << compact(op));
            return mlir::failure();
        }

        L_DEBUG("  Using float execution type " << plan->operandExecutionType << " for " << compact(op));

        const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
        if (!currentResultType) return mlir::failure();

        const std::optional lhsStorage = is_weight(op.getA()) ? target_wgt_type : target_act_type;
        const std::optional rhsStorage = is_weight(op.getB()) ? target_wgt_type : target_act_type;

        const mlir::Value quantisedA = applyMixedPrecisionCast(
            op.getA(), lhsStorage, plan->operandExecutionType, rewriter, op.getLoc()
        );
        const mlir::Value quantisedB = applyMixedPrecisionCast(
            op.getB(), rhsStorage, plan->operandExecutionType, rewriter, op.getLoc()
        );

        const mlir::Value newAZp = ensureType(op->getOperand(2), plan->inputzp.value(), rewriter, op.getLoc());
        const mlir::Value newBZp = ensureType(op->getOperand(3), plan->weightzp.value(), rewriter, op.getLoc());

        const auto newReturnType =
            mlir::RankedTensorType::get(currentResultType.getShape(), plan->resultType);

        auto newMatMul = mlir::tosa::MatMulOp::create(
            rewriter, op.getLoc(), newReturnType, quantisedA, quantisedB, newAZp, newBZp
        );

        newMatMul->setAttrs(op->getAttrs());
        newMatMul->setAttr("conquer.float.quantised", rewriter.getUnitAttr());

        const auto finalOut =
            ensureType(newMatMul.getResult(), currentResultType.getElementType(), rewriter, op.getLoc());
        rewriter.replaceOp(op, finalOut);

        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

    inline void populateComputePatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
        mlir::MLIRContext *context = patterns.getContext();

        patterns.add<QuantiseConv2DPattern>(context, policy);
        patterns.add<QuantiseConv3DPattern>(context, policy);
        patterns.add<QuantiseDepthwiseConv2DPattern>(context, policy);
        patterns.add<QuantiseTransposeConv2DPattern>(context, policy);
        patterns.add<QuantiseMatMulPattern>(context, policy);
    }
}
