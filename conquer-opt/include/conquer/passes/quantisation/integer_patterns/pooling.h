#pragma once

#include "conquer/passes/quantisation/integer_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::integer_quant {

struct QuantiseAvgPool2DPattern : public mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp> {
    QuantiseAvgPool2DPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::AvgPool2dOp op, mlir::PatternRewriter &rewriter) const override {
        L_TRACE("Applying QuantiseAvgPool2DPattern to: " << compact(op));
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        const auto p_opt = extractQuantParams(op, rewriter, quantPolicy);
        if (!p_opt.has_value()) return mlir::failure();
        const auto &p = p_opt.value();

        if (!preflightCheckQuantStats(op, p)) return mlir::failure();

        // 1. Mixed precision fallback for non-integer activation requests.
        if (!p.actIsInt) {
            return handleMixedPrecision(op, rewriter, p);
        }

        const auto plan = planTosaIntMode(op, p.targetActType, get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const mlir::Location loc = op.getLoc();

        // Integer avg_pool2d must remain in a single quant domain.
        // Also, non-i8 signed integer TOSA paths must use zero zero-points.
        const mlir::Type poolElemType = plan->operandExecutionType;
        const bool legalActSym = p.actSym || !poolElemType.isInteger(8);

        // 2. Quantise the input directly into the legal avg_pool execution type.
        const mlir::Value rawInput = op.getInput();
        const auto [quantA, quantInZp] = quantiseActivation(
            rewriter, op, rawInput, poolElemType, poolElemType, legalActSym, p.actCalib
        );
        if (!quantA) return mlir::failure();

        const auto inScales = getScalesFromValue(quantA);
        if (inScales.empty() || inScales[0] <= 0.0) return mlir::failure();

        const double poolScale = inScales[0];

        // Keep the pool output in the same quant domain as the input.
        // For non-i8 integer avgpool, the zero-point must be zero.
        const int64_t poolInZp = poolElemType.isInteger(8) ? quantInZp : 0;
        const int64_t poolOutZp = poolInZp;

        // 3. Construct the integer AvgPool2d.
        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), plan->resultType);

        llvm::SmallVector<mlir::Value> newOperands = {quantA};

        // Some MLIR TOSA versions model input_zp/output_zp as operands.
        if (op.getNumOperands() == 3) {
            const mlir::Type inputZpElemType = plan->inputzp.value_or(poolElemType);
            const mlir::Type outputZpElemType = plan->outputzp.value_or(plan->resultType);

            newOperands.push_back(createZpConst(rewriter, loc, poolInZp, inputZpElemType));
            newOperands.push_back(createZpConst(rewriter, loc, poolOutZp, outputZpElemType));
        }

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "quantization_info" ||
                attrName == "acc_type" ||
                attrName == "input_zp" ||
                attrName == "output_zp")
                continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        // Some older TOSA variants model zero-points as attributes instead.
        if (op.getNumOperands() != 3) {
            if (op->hasAttr("input_zp")) {
                state.addAttribute("input_zp", rewriter.getI64IntegerAttr(poolInZp));
            }
            if (op->hasAttr("output_zp")) {
                state.addAttribute("output_zp", rewriter.getI64IntegerAttr(poolOutZp));
            }
        }

        // Integer avgpool always accumulates in i32.
        state.addAttribute(
            "acc_type",
            mlir::TypeAttr::get(plan->accumulatorType.value_or(rewriter.getI32Type()))
        );
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        auto newOp = rewriter.create(state);

        // 4. Dequantise using the same qparams as the integer avgpool output.
        mlir::OperationState dequantState(loc, "conquer.dequantise");
        dequantState.addOperands(newOp->getResult(0));
        dequantState.addTypes(origResultType);
        dequantState.addAttribute("scales", rewriter.getDenseF64ArrayAttr({poolScale}));
        dequantState.addAttribute("zero_point", rewriter.getI64IntegerAttr(poolOutZp));
        dequantState.addAttribute("partition_kind", rewriter.getStringAttr("per_tensor"));
        dequantState.addAttribute("narrow_range", rewriter.getBoolAttr(legalActSym));

        rewriter.replaceOp(op, rewriter.create(dequantState)->getResult(0));
        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;

    mlir::LogicalResult handleMixedPrecision(mlir::Operation *op, mlir::PatternRewriter &rewriter, const QuantParams &p) const {
        const mlir::Location loc = op->getLoc();

        // This path is only taken when the requested activation type is not integer.
        const mlir::Type requestedFloatType = p.targetActType;
        const auto plan = planRequestedFloatMode<mlir::tosa::AvgPool2dOp>(
            op, requestedFloatType, get_should_squash_acc(op, quantPolicy)
        );
        if (!plan) return mlir::failure();

        llvm::SmallVector<mlir::Value> newOperands;
        newOperands.reserve(op->getNumOperands());

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            mlir::Value operand = op->getOperand(i);

            // Operand 0 is the data tensor; additional operands, when present,
            // are zero-point tensors and must match the execution float type.
            if (i == 0) {
                newOperands.push_back(
                    castToType(rewriter, loc, stripTransientCast(operand), plan->operandExecutionType)
                );
            } else {
                newOperands.push_back(
                    castToType(rewriter, loc, operand, plan->operandExecutionType)
                );
            }
        }

        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), plan->resultType);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "quantization_info" || attrName == "acc_type") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        if (plan->accumulatorType) {
            state.addAttribute("acc_type", mlir::TypeAttr::get(*plan->accumulatorType));
        }
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        auto newOp = rewriter.create(state);
        mlir::Value finalOut = castToType(rewriter, loc, newOp->getResult(0), origResultType.getElementType());

        rewriter.replaceOp(op, finalOut);
        return mlir::success();
    }
};

inline void populatePoolingPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    patterns.add<QuantiseAvgPool2DPattern>(patterns.getContext(), policy);
}

} // namespace conquer::integer_quant
