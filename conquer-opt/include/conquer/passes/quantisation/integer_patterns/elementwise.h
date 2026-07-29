#pragma once

#include "conquer/passes/quantisation/integer_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <algorithm>
#include <optional>

namespace conquer::integer_quant {

[[nodiscard]] inline mlir::Operation *getActivationStatsSource(mlir::Operation *fallbackOp, mlir::Value rawInput) {
    const mlir::Value cleanInput = stripTransientCast(rawInput);
    return cleanInput.getDefiningOp() ? cleanInput.getDefiningOp() : fallbackOp;
}

[[nodiscard]] inline std::optional<double> getPerTensorScaleFromValue(const mlir::Value v) {
    const auto scales = getScalesFromValue(v);
    if (scales.size() != 1) return std::nullopt;
    if (!std::isfinite(scales[0]) || scales[0] <= 0.0) return std::nullopt;
    return scales[0];
}

[[nodiscard]] inline mlir::Value createSameRankScalarConst(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                                           const mlir::RankedTensorType likeType, const int64_t value) {
    llvm::SmallVector<int64_t> shape(likeType.getRank(), 1);
    const auto constType = mlir::RankedTensorType::get(shape, likeType.getElementType());
    const auto elemAttr = rewriter.getIntegerAttr(likeType.getElementType(), value);
    const auto attr = mlir::DenseElementsAttr::get(constType, elemAttr);
    return mlir::tosa::ConstOp::create(rewriter, loc, constType, attr).getResult();
}

[[nodiscard]] inline mlir::Value subtractExactZeroPoint(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                                        const mlir::Value input, const int64_t zp) {
    if (zp == 0) return input;
    const auto inType = llvm::cast<mlir::RankedTensorType>(input.getType());
    const auto zpConst = createSameRankScalarConst(rewriter, loc, inType, zp);
    return mlir::tosa::SubOp::create(rewriter, loc, inType, input, zpConst).getResult();
}

[[nodiscard]] inline bool getLegalOutputQParams(mlir::Operation *op, const QuantParams &p,
                                                double &outScale, int64_t &outZp, bool &outSym) {
    float outMin = 0.0f, outMax = 0.0f;
    if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), outMin, outMax)) {
        return false;
    }

    outSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
    computeScaleAndZPForTosaInt(outMin, outMax, p.targetActType, outSym, outScale, outZp);
    return std::isfinite(outScale) && outScale > 0.0;
}

[[nodiscard]] inline std::optional<double> peekScale(
    mlir::Operation *op, mlir::Value operand, const QuantParams &p, bool isWgt, int operandIdx) {

    if (isWgt) {
        std::vector<float> mins, maxs;
        std::string pk = "per_tensor";
        std::optional<int64_t> axis = std::nullopt;
        const auto reqGran = std::holds_alternative<PerChannelGranularity>(p.wgtGran)
            ? GranularityType::PerChannel : GranularityType::PerTensor;

        if (!getWeightStats(op, operandIdx, p.wgtCalib, reqGran, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis)) {
            if (!getWeightStats(op, operandIdx, p.wgtCalib, GranularityType::PerTensor, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis)) {
                return std::nullopt;
            }
        }
        if (mins.empty() || maxs.empty()) return std::nullopt;
        const bool wSym = getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym);
        double scale = 1.0; int64_t zp = 0;
        computeScaleAndZPForTosaInt(mins[0], maxs[0], p.targetWgtType, wSym, scale, zp);
        return scale;
    } else {
        float min = 0.0f, max = 0.0f;
        mlir::Operation *statsOp = getActivationStatsSource(op, operand);
        if (!getActivationStats(statsOp, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), min, max)) {
            return std::nullopt;
        }
        const bool aSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
        double scale = 1.0; int64_t zp = 0;
        computeScaleAndZPForTosaInt(min, max, p.targetActType, aSym, scale, zp);
        return scale;
    }
}

template <typename OpTy>
struct QuantiseElementwisePatternBase : public mlir::RewritePattern {
    QuantiseElementwisePatternBase(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

protected:
    QuantisationPolicy quantPolicy;

    mlir::LogicalResult handleMixedPrecision(mlir::Operation *op, mlir::PatternRewriter &rewriter, const QuantParams &p) const {
        const mlir::Location loc = op->getLoc();
        const mlir::Type execFloatType =
            p.targetActType.isIntOrIndex() ? mlir::Float32Type::get(rewriter.getContext()) : p.targetActType;

        const auto plan = planRequestedFloatMode<OpTy>(op, execFloatType, get_should_squash_acc(op, quantPolicy));
        if (!plan) return mlir::failure();

        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());

        float outMin = 0.0f, outMax = 0.0f;
        if (p.actIsInt && origResultType.getElementType() != rewriter.getI1Type()) {
            if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), outMin, outMax)) {
                return mlir::failure();
            }
        }

        const bool actSym = p.actIsInt ? getLegalTosaIntSymmetry(p.targetActType, p.actSym) : false;
        const bool wgtSym = p.wgtIsInt ? getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym) : false;

        llvm::SmallVector<mlir::Value> newOperands;
        newOperands.reserve(op->getNumOperands());

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            const mlir::Value operand = op->getOperand(i);

            if (i < 2) {
                if (is_weight(operand)) {
                    if (p.wgtIsInt) {
                        const auto [resValue, _] = quantiseWeight(
                            rewriter, op, i, operand, p.targetWgtType, p.targetWgtType, wgtSym, p.wgtCalib, p.wgtGran
                        );
                        if (!resValue) return mlir::failure();

                        std::vector<float> mins, maxs;
                        std::string pk = "per_tensor";
                        std::optional<int64_t> axis = std::nullopt;

                        const auto reqGran = std::holds_alternative<PerChannelGranularity>(p.wgtGran)
                            ? GranularityType::PerChannel
                            : GranularityType::PerTensor;

                        if (!getWeightStats(op, i, p.wgtCalib, reqGran, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis)) {
                            getWeightStats(op, i, p.wgtCalib, GranularityType::PerTensor, p.targetWgtType.getIntOrFloatBitWidth(), mins, maxs, pk, axis);
                        }

                        if (mins.empty() || maxs.empty()) return mlir::failure();

                        newOperands.push_back(
                            ensureIntType(resValue, execFloatType, rewriter, loc, mins, maxs, wgtSym, pk, axis)
                        );
                    } else {
                        newOperands.push_back(castToType(rewriter, loc, stripTransientCast(operand), execFloatType));
                    }
                } else {
                    if (p.actIsInt) {
                        const auto [resValue, _] = quantiseActivation(
                            rewriter, op, operand, p.targetActType, p.targetActType, actSym, p.actCalib
                        );
                        if (!resValue) return mlir::failure();

                        float aMin = 0.0f, aMax = 0.0f;
                        if (!getActivationStats(
                                getActivationStatsSource(op, operand),
                                p.actCalib,
                                p.targetActType.getIntOrFloatBitWidth(),
                                aMin,
                                aMax)) {
                            return mlir::failure();
                        }

                        newOperands.push_back(
                            ensureIntType(resValue, execFloatType, rewriter, loc, {aMin}, {aMax}, actSym, "per_tensor")
                        );
                    } else {
                        newOperands.push_back(castToType(rewriter, loc, stripTransientCast(operand), execFloatType));
                    }
                }
            } else {
                newOperands.push_back(operand);
            }
        }

        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), plan->resultType);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (attrName == "quantization_info") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        const auto newOp = rewriter.create(state);
        mlir::Value finalOut = castToType(
            rewriter, loc, newOp->getResult(0), p.actIsInt ? origResultType.getElementType() : p.targetActType
        );

        if (p.actIsInt && origResultType.getElementType() != rewriter.getI1Type()) {
            const bool outSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
            const mlir::Value qOut = ensureIntType(
                finalOut, p.targetActType, rewriter, loc, {outMin}, {outMax}, outSym, "per_tensor"
            );
            finalOut = ensureIntType(
                qOut, origResultType.getElementType(), rewriter, loc, {outMin}, {outMax}, outSym, "per_tensor"
            );
        } else {
            finalOut = castToType(rewriter, loc, finalOut, origResultType.getElementType());
        }

        rewriter.replaceOp(op, finalOut);
        return mlir::success();
    }
};

template <typename OpTy>
struct QuantiseScaleAlignedPattern : public QuantiseElementwisePatternBase<OpTy> {
    using QuantiseElementwisePatternBase<OpTy>::QuantiseElementwisePatternBase;

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();
        if (!should_quantise(op, this->quantPolicy)) return mlir::failure();

        const auto pOpt = extractQuantParams(op, rewriter, this->quantPolicy);
        if (!pOpt.has_value()) return mlir::failure();

        const auto &p = pOpt.value();
        if (!preflightCheckQuantStats(op, p)) return mlir::failure();
        if (!p.actIsInt) return this->handleMixedPrecision(op, rewriter, p);
        if (std::holds_alternative<PerChannelGranularity>(p.wgtGran)) return mlir::failure();

        const auto plan = planTosaIntMode(op, p.targetActType, get_should_squash_acc(op, this->quantPolicy));
        if (!plan) return mlir::failure();

        // EARLY MATH VERIFICATION - SAFE BAILOUT BEFORE CREATING IR
        const auto sAOpt = peekScale(op, op->getOperand(0), p, is_weight(op->getOperand(0)), 0);
        const auto sBOpt = peekScale(op, op->getOperand(1), p, is_weight(op->getOperand(1)), 1);
        if (!sAOpt || !sBOpt) return mlir::failure();

        double outScale = 1.0;
        int64_t outZp = 0;
        bool outSym = false;
        if (!getLegalOutputQParams(op, p, outScale, outZp, outSym)) return mlir::failure();

        int32_t multA = 0, multB = 0;
        int8_t shiftA = 0, shiftB = 0;
        if (!tryComputeMultiplierAndShift32(*sAOpt / outScale, multA, shiftA)) return mlir::failure();
        if (!tryComputeMultiplierAndShift32(*sBOpt / outScale, multB, shiftB)) return mlir::failure();

        L_INFO("Quantising " << compact(op) << " to integer.");

        const bool actSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
        const bool wgtSym = getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym);

        auto getQuantisedOperand = [&](mlir::Value operand, int index) -> std::pair<mlir::Value, int64_t> {
            if (is_weight(operand)) {
                return quantiseWeight(rewriter, op, index, operand, p.targetWgtType, p.targetWgtType, wgtSym, p.wgtCalib, p.wgtGran);
            }
            return quantiseActivation(rewriter, op, operand, p.targetActType, p.targetActType, actSym, p.actCalib);
        };

        const auto [quantA, aZpVal] = getQuantisedOperand(op->getOperand(0), 0);
        const auto [quantB, bZpVal] = getQuantisedOperand(op->getOperand(1), 1);
        if (!quantA || !quantB) return mlir::failure();

        const mlir::Type i32Type = rewriter.getI32Type();
        const mlir::Location loc = op->getLoc();

        const mlir::Value rescaledA = createTosaRescale(rewriter, loc, quantA, multA, shiftA, aZpVal, 0, i32Type);
        const mlir::Value rescaledB = createTosaRescale(rewriter, loc, quantB, multB, shiftB, bZpVal, 0, i32Type);

        const auto outShape = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
        const auto resType = mlir::RankedTensorType::get(outShape, i32Type);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands({rescaledA, rescaledB});
        state.addTypes(resType);

        for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue() == "quantization_info") continue;
            if (attr.getName().getValue() == "nan_mode") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());
        auto newOp = rewriter.create(state);

        int32_t unityMult = 0;
        int8_t unityShift = 0;
        if (!tryComputeMultiplierAndShift32(1.0, unityMult, unityShift)) return mlir::failure();

        const mlir::Value finalIntOut = createTosaRescale(
            rewriter, loc, newOp->getResult(0), unityMult, unityShift, 0, outZp, p.targetActType
        );

        mlir::OperationState dequantState(loc, "conquer.dequantise");
        dequantState.addOperands(finalIntOut);
        dequantState.addTypes(op->getResult(0).getType());
        dequantState.addAttribute("scales", rewriter.getDenseF64ArrayAttr({outScale}));
        dequantState.addAttribute("zero_point", rewriter.getI64IntegerAttr(outZp));
        dequantState.addAttribute("partition_kind", rewriter.getStringAttr("per_tensor"));
        dequantState.addAttribute("narrow_range", rewriter.getBoolAttr(outSym));

        rewriter.replaceOp(op, rewriter.create(dequantState)->getResult(0));
        return mlir::success();
    }
};

template <typename OpTy>
struct QuantiseMulPattern : public QuantiseElementwisePatternBase<OpTy> {
    using QuantiseElementwisePatternBase<OpTy>::QuantiseElementwisePatternBase;

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();
        if (!should_quantise(op, this->quantPolicy)) return mlir::failure();

        const auto pOpt = extractQuantParams(op, rewriter, this->quantPolicy);
        if (!pOpt.has_value()) return mlir::failure();

        const auto &p = pOpt.value();
        if (!preflightCheckQuantStats(op, p)) return mlir::failure();
        if (!p.actIsInt) return this->handleMixedPrecision(op, rewriter, p);
        if (std::holds_alternative<PerChannelGranularity>(p.wgtGran)) return mlir::failure();

        const auto plan = planTosaIntMode(op, p.targetActType, get_should_squash_acc(op, this->quantPolicy));
        if (!plan) return mlir::failure();

        // EARLY MATH VERIFICATION - SAFE BAILOUT BEFORE CREATING IR
        const auto sAOpt = peekScale(op, op->getOperand(0), p, is_weight(op->getOperand(0)), 0);
        const auto sBOpt = peekScale(op, op->getOperand(1), p, is_weight(op->getOperand(1)), 1);
        if (!sAOpt || !sBOpt) return mlir::failure();

        double outScale = 1.0;
        int64_t outZp = 0;
        bool outSym = false;
        if (!getLegalOutputQParams(op, p, outScale, outZp, outSym)) return mlir::failure();

        int32_t mult = 0;
        int8_t shift = 0;
        if (!tryComputeMultiplierAndShift32((*sAOpt * *sBOpt) / outScale, mult, shift)) return mlir::failure();

        L_INFO("Quantising " << compact(op) << " to integer.");

        const bool actSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
        const bool wgtSym = getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym);

        auto getQuantisedOperand = [&](mlir::Value operand, int index) -> std::pair<mlir::Value, int64_t> {
            if (is_weight(operand)) {
                return quantiseWeight(rewriter, op, index, operand, p.targetWgtType, p.targetWgtType, wgtSym, p.wgtCalib, p.wgtGran);
            }
            return quantiseActivation(rewriter, op, operand, p.targetActType, p.targetActType, actSym, p.actCalib);
        };

        const auto [quantA, aZpVal] = getQuantisedOperand(op->getOperand(0), 0);
        const auto [quantB, bZpVal] = getQuantisedOperand(op->getOperand(1), 1);
        if (!quantA || !quantB) return mlir::failure();

        const mlir::Type i32Type = rewriter.getI32Type();
        const mlir::Location loc = op->getLoc();

        const mlir::Value zeroCenteredA = subtractExactZeroPoint(
            rewriter, loc, castToType(rewriter, loc, quantA, i32Type), aZpVal
        );
        const mlir::Value zeroCenteredB = subtractExactZeroPoint(
            rewriter, loc, castToType(rewriter, loc, quantB, i32Type), bZpVal
        );

        const auto outShape = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
        const auto resType = mlir::RankedTensorType::get(outShape, i32Type);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands({zeroCenteredA, zeroCenteredB});

        const auto shiftZeroType = mlir::RankedTensorType::get({1}, rewriter.getI8Type());
        const auto shiftZeroAttr = rewriter.getIntegerAttr(rewriter.getI8Type(), 0);
        const auto shiftZero = mlir::tosa::ConstOp::create(
            rewriter, loc, shiftZeroType, mlir::DenseElementsAttr::get(shiftZeroType, shiftZeroAttr)
        ).getResult();

        state.addOperands({shiftZero});
        state.addTypes(resType);

        for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue() == "quantization_info") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());
        auto mulOp = rewriter.create(state);

        const mlir::Value finalIntOut = createTosaRescale(
            rewriter, loc, mulOp->getResult(0), mult, shift, 0, outZp, p.targetActType
        );

        mlir::OperationState dequantState(loc, "conquer.dequantise");
        dequantState.addOperands(finalIntOut);
        dequantState.addTypes(op->getResult(0).getType());
        dequantState.addAttribute("scales", rewriter.getDenseF64ArrayAttr({outScale}));
        dequantState.addAttribute("zero_point", rewriter.getI64IntegerAttr(outZp));
        dequantState.addAttribute("partition_kind", rewriter.getStringAttr("per_tensor"));
        dequantState.addAttribute("narrow_range", rewriter.getBoolAttr(outSym));

        rewriter.replaceOp(op, rewriter.create(dequantState)->getResult(0));
        return mlir::success();
    }
};

template <typename OpTy>
struct QuantiseComparatorPattern : public QuantiseElementwisePatternBase<OpTy> {
    using QuantiseElementwisePatternBase<OpTy>::QuantiseElementwisePatternBase;

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();
        if (!should_quantise(op, this->quantPolicy)) return mlir::failure();

        const auto pOpt = extractQuantParams(op, rewriter, this->quantPolicy);
        if (!pOpt.has_value()) return mlir::failure();

        const auto &p = pOpt.value();
        if (!preflightCheckQuantStats(op, p)) return mlir::failure();
        if (!p.actIsInt) return this->handleMixedPrecision(op, rewriter, p);
        if (std::holds_alternative<PerChannelGranularity>(p.wgtGran)) return mlir::failure();

        const auto plan = planTosaIntMode(op, p.targetActType, get_should_squash_acc(op, this->quantPolicy));
        if (!plan) return mlir::failure();

        const auto sAOpt = peekScale(op, op->getOperand(0), p, is_weight(op->getOperand(0)), 0);
        const auto sBOpt = peekScale(op, op->getOperand(1), p, is_weight(op->getOperand(1)), 1);
        if (!sAOpt || !sBOpt) return mlir::failure();

        const double sMin = std::min(*sAOpt, *sBOpt);
        if (*sAOpt > sMin * 1.0001) {
            int32_t mult = 0; int8_t shift = 0;
            if (!tryComputeMultiplierAndShift32(*sAOpt / sMin, mult, shift)) return mlir::failure();
        }
        if (*sBOpt > sMin * 1.0001) {
            int32_t mult = 0; int8_t shift = 0;
            if (!tryComputeMultiplierAndShift32(*sBOpt / sMin, mult, shift)) return mlir::failure();
        }

        L_INFO("Quantising " << compact(op) << " to integer.");

        const bool actSym = getLegalTosaIntSymmetry(p.targetActType, p.actSym);
        const bool wgtSym = getLegalTosaIntSymmetry(p.targetWgtType, p.wgtSym);

        auto getQuantisedOperand = [&](mlir::Value operand, int index) -> std::pair<mlir::Value, int64_t> {
            if (is_weight(operand)) {
                return quantiseWeight(rewriter, op, index, operand, p.targetWgtType, p.targetWgtType, wgtSym, p.wgtCalib, p.wgtGran);
            }
            return quantiseActivation(rewriter, op, operand, p.targetActType, p.targetActType, actSym, p.actCalib);
        };

        const auto [quantA, aZpVal] = getQuantisedOperand(op->getOperand(0), 0);
        const auto [quantB, bZpVal] = getQuantisedOperand(op->getOperand(1), 1);
        if (!quantA || !quantB) return mlir::failure();

        const mlir::Type i32Type = rewriter.getI32Type();
        const mlir::Location loc = op->getLoc();

        mlir::Value rescaledA;
        if (*sAOpt > sMin * 1.0001) {
            int32_t mult = 0;
            int8_t shift = 0;
            if (!tryComputeMultiplierAndShift32(*sAOpt / sMin, mult, shift)) return mlir::failure();
            rescaledA = createTosaRescale(rewriter, loc, quantA, mult, shift, aZpVal, 0, i32Type);
        } else {
            rescaledA = subtractExactZeroPoint(rewriter, loc, castToType(rewriter, loc, quantA, i32Type), aZpVal);
        }

        mlir::Value rescaledB;
        if (*sBOpt > sMin * 1.0001) {
            int32_t mult = 0;
            int8_t shift = 0;
            if (!tryComputeMultiplierAndShift32(*sBOpt / sMin, mult, shift)) return mlir::failure();
            rescaledB = createTosaRescale(rewriter, loc, quantB, mult, shift, bZpVal, 0, i32Type);
        } else {
            rescaledB = subtractExactZeroPoint(rewriter, loc, castToType(rewriter, loc, quantB, i32Type), bZpVal);
        }

        const auto outShape = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
        const auto resType = mlir::RankedTensorType::get(outShape, rewriter.getI1Type());

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands({rescaledA, rescaledB});
        state.addTypes(resType);

        for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue() == "quantization_info") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }

        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());
        rewriter.replaceOp(op, rewriter.create(state)->getResult(0));
        return mlir::success();
    }
};

inline void populateElementWisePatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    // Group 1: Scale-aligned
    patterns.add<QuantiseScaleAlignedPattern<mlir::tosa::AddOp>>(context, policy);
    patterns.add<QuantiseScaleAlignedPattern<mlir::tosa::SubOp>>(context, policy);
    patterns.add<QuantiseScaleAlignedPattern<mlir::tosa::MaximumOp>>(context, policy);
    patterns.add<QuantiseScaleAlignedPattern<mlir::tosa::MinimumOp>>(context, policy);

    // Group 2: Scale-multiplied
    patterns.add<QuantiseMulPattern<mlir::tosa::MulOp>>(context, policy);

    // Group 3: Comparators
    patterns.add<QuantiseComparatorPattern<mlir::tosa::EqualOp>>(context, policy);
    patterns.add<QuantiseComparatorPattern<mlir::tosa::GreaterOp>>(context, policy);
    patterns.add<QuantiseComparatorPattern<mlir::tosa::GreaterEqualOp>>(context, policy);
}

} // namespace conquer::integer_quant