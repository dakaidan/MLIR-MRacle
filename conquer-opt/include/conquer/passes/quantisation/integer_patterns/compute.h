#pragma once

#include "conquer/passes/quantisation/integer_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <type_traits>

namespace conquer::integer_quant {

template <typename OpTy>
struct QuantiseDotProductPattern : public mlir::RewritePattern {
    QuantiseDotProductPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        L_INFO("Quantising " << compact(op) << " to integer.");

        const auto p = extractQuantParams(op, rewriter, quantPolicy);
        if (!p.has_value()) {
            L_DEBUG("  Failed to extract quantisation parameters for " << compact(op));
            return mlir::failure();
        }

        if (!preflightCheckQuantStats(op, *p)) {
             L_DEBUG("  Quantisation pre-flight check failed for " << compact(op));
             return mlir::failure();
        }

        if (p->wgtIsInt && p->actIsInt) {
            L_DEBUG("  Using pure integer path for " << compact(op));
            return handlePureInteger(op, rewriter, *p);
        }
        L_DEBUG("  Using mixed precision path for " << compact(op));
        return handleMixedPrecision(op, rewriter, *p);
    }

private:
    QuantisationPolicy quantPolicy;
    static constexpr bool is_matmul = std::is_same_v<OpTy, mlir::tosa::MatMulOp>;

    mlir::LogicalResult handlePureInteger(mlir::Operation *op, mlir::PatternRewriter &rewriter, const QuantParams &p) const {
        const auto plan = planTosaIntMixedMode(
            op, p.targetActType, p.targetWgtType, get_should_squash_acc(op, quantPolicy)
        );
        if (!plan) return mlir::failure();

        const bool actSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
        const bool wgtSym = getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym);

        // 1. Quantise A (input) and B (weight)
        const std::pair<mlir::Value, int64_t> AVal = quantiseActivation(
            rewriter, op, op->getOperand(0), p.targetActType, plan->operandExecutionType, actSym, p.actCalib
        );
        const auto [quantA, aZpVal] = AVal;
        if (!quantA) return mlir::failure();

        mlir::Type wgtComputeType;
        std::pair<mlir::Value, int64_t> BVal;
        if (is_matmul && !is_weight(op->getOperand(1))) {
            wgtComputeType = plan->inputzp.value_or(plan->operandExecutionType);
            BVal = quantiseActivation(
                rewriter, op, op->getOperand(1), p.targetActType, wgtComputeType, actSym, p.actCalib
            );
        } else {
            wgtComputeType = plan->weightzp.value_or(plan->operandExecutionType);
            BVal = quantiseWeight(
                rewriter, op, 1, op->getOperand(1), p.targetWgtType, wgtComputeType, wgtSym, p.wgtCalib, p.wgtGran
            );
        }
        const auto [quantB, bZpVal] = BVal;
        if (!quantB) return mlir::failure();

        const auto scaleA = getScalesFromValue(quantA);
        const auto scaleB = getScalesFromValue(quantB);
        const mlir::Type accType = plan->resultType;

        llvm::SmallVector<mlir::Value> newOperands = {quantA, quantB};

        // 2. Bias for conv-family ops
        if constexpr (!is_matmul) {
            const size_t outChannels = scaleB.size();
            std::vector<double> biasScales(outChannels);
            for (size_t i = 0; i < outChannels; ++i) {
                const double sA = scaleA.size() > 1 ? scaleA[i] : scaleA[0];
                const double sB = scaleB.size() > 1 ? scaleB[i] : scaleB[0];
                biasScales[i] = sA * sB;
            }

            const mlir::Value quantBias = quantiseBias(rewriter, op->getLoc(), op->getOperand(2), accType, biasScales);
            newOperands.push_back(quantBias);
        }

        // 3. Scalar zero-point operands
        const mlir::Type inputZpType = plan->inputzp.value_or(plan->operandExecutionType);
        const mlir::Type weightZpType = plan->weightzp.value_or(wgtComputeType);
        newOperands.push_back(createZpConst(rewriter, op->getLoc(), aZpVal, inputZpType));
        newOperands.push_back(createZpConst(rewriter, op->getLoc(), bZpVal, weightZpType));

        // 4. Emit integer op
        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), accType);

        mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "quantization_info" || attrName == "acc_type") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        if constexpr (!is_matmul) {
            if (plan->accumulatorType) {
                state.addAttribute("acc_type", mlir::TypeAttr::get(*plan->accumulatorType));
            }
        }

        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());
        auto newOp = rewriter.create(state);

        // 5. Rescale and dequantise output
        const mlir::Value finalOut = emitRescaleAndDequantise(
            rewriter, op, newOp->getResult(0), p.targetActType, scaleA, scaleB, actSym, p.actCalib
        );
        if (!finalOut) return mlir::failure();

        rewriter.replaceOp(op, finalOut);
        return mlir::success();
    }

    mlir::LogicalResult handleMixedPrecision(mlir::Operation *op, mlir::PatternRewriter &rewriter, const QuantParams &p) const {
        const mlir::Location loc = op->getLoc();
        const mlir::Type execFloatType = p.actIsInt ? p.targetWgtType : p.targetActType;

        const auto plan = planTosaFloatMode(op, execFloatType, get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const bool actSym = p.actIsInt ? getLegalTosaIntSymmetry(p.targetActType, p.actSym) : false;
        const bool wgtSym = p.wgtIsInt ? getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym) : false;

        mlir::Value finalA;
        mlir::Value finalB;

        // 1. Prepare activation
        if (p.actIsInt) {
            const mlir::Value rawA = op->getOperand(0);
            const auto [resAValue, _] = quantiseActivation(
                rewriter, op, rawA, p.targetActType, p.targetActType, actSym, p.actCalib
            );
            if (!resAValue) return mlir::failure();

            const mlir::Value cleanA = stripTransientCast(rawA);
            mlir::Operation *aStatsOp = cleanA.getDefiningOp() ? cleanA.getDefiningOp() : op;

            float aMin = 0.0f, aMax = 0.0f;
            if (!getActivationStats(aStatsOp, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), aMin, aMax)) {
                return mlir::failure();
            }

            finalA = ensureIntType(resAValue, execFloatType, rewriter, loc, {aMin}, {aMax}, actSym, "per_tensor");
        } else {
            finalA = castToType(rewriter, loc, stripTransientCast(op->getOperand(0)), execFloatType);
        }

        // 2. Prepare Operand 1 (Weight or Activation)
        if (is_matmul && !is_weight(op->getOperand(1))) {
            // It's a MatMul where Operand 1 is an Activation (e.g., Attention QxK^T)
            if (p.actIsInt) {
                const mlir::Value rawB = op->getOperand(1);
                const auto [resBValue, _] = quantiseActivation(
                    rewriter, op, rawB, p.targetActType, p.targetActType, actSym, p.actCalib
                );
                if (!resBValue) return mlir::failure();

                const mlir::Value cleanB = stripTransientCast(rawB);
                mlir::Operation *bStatsOp = cleanB.getDefiningOp() ? cleanB.getDefiningOp() : op;

                float bMin = 0.0f, bMax = 0.0f;
                if (!getActivationStats(bStatsOp, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), bMin, bMax)) {
                    return mlir::failure();
                }

                finalB = ensureIntType(resBValue, execFloatType, rewriter, loc, {bMin}, {bMax}, actSym, "per_tensor");
            } else {
                finalB = castToType(rewriter, loc, stripTransientCast(op->getOperand(1)), execFloatType);
            }
        } else {
            // Standard Weight processing
            if (p.wgtIsInt) {
                const auto [resBValue, _] = quantiseWeight(
                    rewriter, op, 1, op->getOperand(1), p.targetWgtType, p.targetWgtType, wgtSym, p.wgtCalib, p.wgtGran
                );
                if (!resBValue) return mlir::failure();

                std::vector<float> mins, maxs;
                std::string pk = "per_tensor";
                std::optional<int64_t> axis = std::nullopt;
                const auto reqGran = std::holds_alternative<PerChannelGranularity>(p.wgtGran)
                    ? GranularityType::PerChannel
                    : GranularityType::PerTensor;

                if (!getWeightStats(op, 1, p.wgtCalib, reqGran, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis)) {
                    getWeightStats(op, 1, p.wgtCalib, GranularityType::PerTensor, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis);
                }

                finalB = ensureIntType(resBValue, execFloatType, rewriter, loc, mins, maxs, wgtSym, pk, axis);
            } else {
                finalB = castToType(rewriter, loc, stripTransientCast(op->getOperand(1)), execFloatType);
            }
        }

        // 3. Assemble operands
        llvm::SmallVector<mlir::Value> newOperands = {finalA, finalB};

        if constexpr (!is_matmul) {
            newOperands.push_back(castToType(rewriter, loc, stripTransientCast(op->getOperand(2)), execFloatType));
        }

        const mlir::Value floatZp = getFloatZeroPointConst(rewriter, loc, execFloatType);
        if (!floatZp) return mlir::failure();

        newOperands.push_back(floatZp);
        newOperands.push_back(floatZp);

        // 4. Emit float op
        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        const mlir::Type resultType = plan->resultType;
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), resultType);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "quantization_info" || attrName == "acc_type") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        if constexpr (!is_matmul) {
            if (plan->accumulatorType) {
                state.addAttribute("acc_type", mlir::TypeAttr::get(*plan->accumulatorType));
            }
        }

        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        auto newOp = rewriter.create(state);
        mlir::Value finalOut = castToType(
            rewriter, loc, newOp->getResult(0),
            p.actIsInt ? origResultType.getElementType() : p.targetActType
        );

        // 5. Re-quantise activation output if requested
        if (p.actIsInt) {
            float outMin = 0.0f, outMax = 0.0f;
            if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), outMin, outMax)) {
                return mlir::failure();
            }

            const mlir::Value qOut = ensureIntType(
                finalOut, p.targetActType, rewriter, loc, {outMin}, {outMax}, actSym, "per_tensor"
            );
            finalOut = ensureIntType(
                qOut, origResultType.getElementType(), rewriter, loc, {outMin}, {outMax}, actSym, "per_tensor"
            );
        } else {
            finalOut = castToType(rewriter, loc, finalOut, origResultType.getElementType());
        }

        rewriter.replaceOp(op, finalOut);
        return mlir::success();
    }
};

using QuantiseMatMulPattern = QuantiseDotProductPattern<mlir::tosa::MatMulOp>;
using QuantiseConv2DPattern = QuantiseDotProductPattern<mlir::tosa::Conv2DOp>;
using QuantiseConv3DPattern = QuantiseDotProductPattern<mlir::tosa::Conv3DOp>;
using QuantiseDepthwiseConv2DPattern = QuantiseDotProductPattern<mlir::tosa::DepthwiseConv2DOp>;
using QuantiseTransposeConv2DPattern = QuantiseDotProductPattern<mlir::tosa::TransposeConv2DOp>;

inline void populateComputePatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();
    patterns.add<QuantiseMatMulPattern>(context, policy);
    patterns.add<QuantiseConv2DPattern>(context, policy);
    patterns.add<QuantiseConv3DPattern>(context, policy);
    patterns.add<QuantiseDepthwiseConv2DPattern>(context, policy);
    patterns.add<QuantiseTransposeConv2DPattern>(context, policy);
}

} // namespace conquer::integer_quant
