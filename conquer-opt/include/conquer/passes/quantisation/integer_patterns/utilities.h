#pragma once

#include "conquer/passes/quantisation/shared/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/dialect/ConquerDialect.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace conquer {

[[nodiscard]] inline bool is_activation_symmetric(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto cfg = policy.get_config(getOpName(op)); cfg && cfg->activation_policy) {
        return cfg->activation_policy->scheme == Scheme::Symmetric;
    }
    return false;
}

[[nodiscard]] inline bool is_weight_symmetric(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto cfg = policy.get_config(getOpName(op)); cfg && cfg->weight_policy) {
        return cfg->weight_policy->scheme == Scheme::Symmetric;
    }
    return true;
}

[[nodiscard]] inline CalibrationMethod get_activation_calibration_method(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto cfg = policy.get_config(getOpName(op)); cfg && cfg->activation_policy) {
        return cfg->activation_policy->calibration;
    }
    return CalibrationMethod::MinMax;
}

[[nodiscard]] inline CalibrationMethod get_weight_calibration_method(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto cfg = policy.get_config(getOpName(op)); cfg && cfg->weight_policy) {
        return cfg->weight_policy->calibration;
    }
    return CalibrationMethod::MinMax;
}

[[nodiscard]] inline Granularity get_weight_granularity(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto cfg = policy.get_config(getOpName(op)); cfg && cfg->weight_policy) {
        return cfg->weight_policy->granularity;
    }
    return PerTensorGranularity{};
}

[[nodiscard]] inline std::string getStatPrefix(const CalibrationMethod method, const unsigned bitWidth) {
    switch (method) {
        case CalibrationMethod::MinMax:     return "min_max.";
        case CalibrationMethod::Percentile: return "percentile.";
        case CalibrationMethod::Entropy:    return (bitWidth == 4) ? "kl.int4." : "kl.int8.";
    }
    return "min_max.";
}

inline bool getActivationStats(mlir::Operation *op, const CalibrationMethod method, const unsigned targetBitWidth,
                               float &min, float &max) {
    const std::string attrPrefix = "activation_stats." + getStatPrefix(method, targetBitWidth);

    const auto minAttr = op->getAttrOfType<mlir::FloatAttr>(attrPrefix + "min");
    const auto maxAttr = op->getAttrOfType<mlir::FloatAttr>(attrPrefix + "max");

    if (!minAttr || !maxAttr) return false;

    min = static_cast<float>(minAttr.getValue().convertToDouble());
    max = static_cast<float>(maxAttr.getValue().convertToDouble());
    return true;
}

inline bool getWeightStats(mlir::Operation *consumerOp, const unsigned operandIndex,
                           const CalibrationMethod method, const GranularityType granularity,
                           const unsigned targetBitWidth, std::vector<float> &mins,
                           std::vector<float> &maxs, std::string &partition_kind,
                           std::optional<int64_t> &axis) {
    const std::string basePrefix = "weight_use." + std::to_string(operandIndex) + ".";

    if (const auto pkAttr = consumerOp->getAttrOfType<mlir::StringAttr>(basePrefix + "partition_kind")) {
        partition_kind = pkAttr.getValue().str();
    }
    if (const auto axisAttr = consumerOp->getAttrOfType<mlir::IntegerAttr>(basePrefix + "axis")) {
        axis = axisAttr.getInt();
    }

    const std::string statPrefix = basePrefix + getStatPrefix(method, targetBitWidth);

    if (granularity == GranularityType::PerTensor) {
        const auto minAttr = consumerOp->getAttrOfType<mlir::FloatAttr>(statPrefix + "per_tensor.min");
        const auto maxAttr = consumerOp->getAttrOfType<mlir::FloatAttr>(statPrefix + "per_tensor.max");
        if (!minAttr || !maxAttr) return false;

        mins = {static_cast<float>(minAttr.getValue().convertToDouble())};
        maxs = {static_cast<float>(maxAttr.getValue().convertToDouble())};
        partition_kind = "per_tensor";
        axis = std::nullopt;
        return true;
    }

    const auto minsAttr = consumerOp->getAttrOfType<mlir::ArrayAttr>(statPrefix + "per_channel.mins");
    const auto maxsAttr = consumerOp->getAttrOfType<mlir::ArrayAttr>(statPrefix + "per_channel.maxes");
    if (!minsAttr || !maxsAttr) return false;

    mins.clear();
    maxs.clear();

    for (const auto m : minsAttr) {
        mins.push_back(static_cast<float>(llvm::cast<mlir::FloatAttr>(m).getValue().convertToDouble()));
    }
    for (const auto m : maxsAttr) {
        maxs.push_back(static_cast<float>(llvm::cast<mlir::FloatAttr>(m).getValue().convertToDouble()));
    }
    return true;
}

// -----------------------------------------------------------------------------
// TOSA integer legality helpers
// -----------------------------------------------------------------------------

[[nodiscard]] inline bool requiresZeroZeroPointForTosaInt(const mlir::Type t) {
    return t.isIntOrIndex() && !t.isInteger(8);
}

[[nodiscard]] inline int64_t getLegalTosaIntZeroPoint(const mlir::Type t, const int64_t requestedZp) {
    if (!t.isIntOrIndex()) return requestedZp;
    return t.isInteger(8) ? requestedZp : 0;
}

[[nodiscard]] inline bool getLegalTosaIntSymmetry(const mlir::Type t, const bool requestedSymmetric) {
    if (!t.isIntOrIndex()) return requestedSymmetric;
    return requestedSymmetric || requiresZeroZeroPointForTosaInt(t);
}

[[nodiscard]] inline bool requiresSymmetricPartitionedIntQuant(const std::string &partitionKind) {
    // conquer.quantise / conquer.dequantise only carry a scalar zero-point.
    // So any non-per-tensor partitioned integer quantisation must be symmetric.
    return partitionKind != "per_tensor";
}

// -----------------------------------------------------------------------------
// Quant param computation
// -----------------------------------------------------------------------------

inline void computeScaleAndZP(const float min, const float max, const unsigned bitWidth, const bool symmetric,
                              double &scale, int64_t &zp) {
    const int64_t qMin = symmetric ? -(1LL << (bitWidth - 1)) + 1 : -(1LL << (bitWidth - 1));
    const int64_t qMax = (1LL << (bitWidth - 1)) - 1;

    if (symmetric) {
        const float absMax = std::max(std::abs(min), std::abs(max));
        scale = static_cast<double>(absMax) / static_cast<double>(qMax);
        zp = 0;
    } else {
        scale = static_cast<double>(max - min) / static_cast<double>(qMax - qMin);
        if (scale == 0.0) {
            scale = 1.0;
            zp = 0;
            return;
        }

        const double rawZp = static_cast<double>(qMin) - (static_cast<double>(min) / scale);
        zp = static_cast<int64_t>(std::round(rawZp));
        zp = std::clamp(zp, qMin, qMax);
    }

    if (scale == 0.0) scale = 1.0;
}

inline void computeScaleAndZPForTosaInt(const float min, const float max, const mlir::Type storageType,
                                        const bool requestedSymmetric, double &scale, int64_t &zp) {
    const unsigned bitWidth = storageType.getIntOrFloatBitWidth();
    const bool effectiveSymmetric = getLegalTosaIntSymmetry(storageType, requestedSymmetric);
    computeScaleAndZP(min, max, bitWidth, effectiveSymmetric, scale, zp);
    zp = getLegalTosaIntZeroPoint(storageType, zp);
}

// -----------------------------------------------------------------------------
// TOSA rescale multiplier helpers
// -----------------------------------------------------------------------------

[[nodiscard]] inline bool isValidTosaRescaleShift(const int32_t shift) {
    // TOSA rescale uses int8 shift operands; the semantic range is bounded.
    // We reject values outside the standard legal range instead of silently
    // clamping to a different numerical transform.
    return shift >= 0 && shift <= 62;
}

[[nodiscard]] inline bool tryComputeMultiplierAndShift32(const double scale, int32_t &multiplier, int8_t &shift) {
    if (scale <= 0.0 || !std::isfinite(scale)) {
        multiplier = 0;
        shift = 0;
        return false;
    }

    int exp;
    const double mantissa = std::frexp(scale, &exp);
    auto qFixed = static_cast<int64_t>(std::round(mantissa * static_cast<double>(1ll << 31)));

    if (qFixed == (1ll << 31)) {
        qFixed /= 2;
        ++exp;
    }

    const auto computedShift = static_cast<int32_t>(31 - exp);
    if (qFixed <= 0 || qFixed > std::numeric_limits<int32_t>::max() || !isValidTosaRescaleShift(computedShift)) {
        multiplier = 0;
        shift = 0;
        return false;
    }

    multiplier = static_cast<int32_t>(qFixed);
    shift = static_cast<int8_t>(computedShift);
    return true;
}

[[nodiscard]] inline bool tryComputeMultiplierAndShift16(const double scale, int16_t &multiplier, int8_t &shift) {
    if (scale <= 0.0 || !std::isfinite(scale)) {
        multiplier = 0;
        shift = 0;
        return false;
    }

    int exp;
    const double mantissa = std::frexp(scale, &exp);
    auto qFixed = static_cast<int64_t>(std::round(mantissa * static_cast<double>(1ll << 15)));

    if (qFixed == (1ll << 15)) {
        qFixed /= 2;
        ++exp;
    }

    const auto computedShift = static_cast<int32_t>(15 - exp);
    if (qFixed <= 0 || qFixed > std::numeric_limits<int16_t>::max() || !isValidTosaRescaleShift(computedShift)) {
        multiplier = 0;
        shift = 0;
        return false;
    }

    multiplier = static_cast<int16_t>(qFixed);
    shift = static_cast<int8_t>(computedShift);
    return true;
}

// Backwards-compatible wrapper for existing call sites that still expect the
// old signature. New code should prefer tryComputeMultiplierAndShift32/16.
inline void computeMultiplierAndShift(const double scale, int32_t &multiplier, int32_t &shift) {
    int8_t localShift = 0;
    if (!tryComputeMultiplierAndShift32(scale, multiplier, localShift)) {
        multiplier = 0;
        shift = 0;
        return;
    }
    shift = static_cast<int32_t>(localShift);
}

// -----------------------------------------------------------------------------
// General helpers
// -----------------------------------------------------------------------------

[[nodiscard]] inline mlir::Value stripTransientCast(const mlir::Value v) {
    mlir::Value currentVal = v;
    while (auto castOp = currentVal.getDefiningOp<mlir::tosa::CastOp>()) {
        if (castOp->hasAttr("conquer.int.transient") || castOp->hasAttr("conquer.cast") || castOp->hasAttr("conquer.bridge")) {
            currentVal = castOp.getInput();
        } else {
            break;
        }
    }
    return currentVal;
}

[[nodiscard]] inline std::vector<double> getScalesFromValue(const mlir::Value v) {
    mlir::Operation *op = v.getDefiningOp();
    if (auto castOp = llvm::dyn_cast_or_null<mlir::tosa::CastOp>(op)) {
        op = castOp.getInput().getDefiningOp();
    }
    if (op && op->hasAttr("scales")) {
        const auto attr = op->getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
        return {attr.asArrayRef().begin(), attr.asArrayRef().end()};
    }
    return {1.0};
}

[[nodiscard]] inline mlir::Value createZpConst(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                               const int64_t zp, const mlir::Type zpType) {
    const int64_t legalZp = getLegalTosaIntZeroPoint(zpType, zp);
    const auto tensorType = mlir::RankedTensorType::get({1}, zpType);
    const auto elemAttr = rewriter.getIntegerAttr(zpType, legalZp);
    const auto attr = mlir::DenseElementsAttr::get(tensorType, elemAttr);
    return mlir::tosa::ConstOp::create(rewriter, loc, tensorType, attr).getResult();
}

[[nodiscard]] inline mlir::Value getFloatZeroPointConst(mlir::PatternRewriter &rewriter,
                                                        const mlir::Location loc,
                                                        const mlir::Type floatType) {
    const auto zpType = mlir::RankedTensorType::get({1}, floatType);
    const auto attr = llvm::cast<mlir::DenseElementsAttr>(rewriter.getZeroAttr(zpType));
    return mlir::tosa::ConstOp::create(rewriter, loc, zpType, attr).getResult();
}

[[nodiscard]] inline mlir::Value castToType(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                            const mlir::Value v, const mlir::Type t) {
    const auto vType = llvm::cast<mlir::RankedTensorType>(v.getType());
    if (vType.getElementType() == t) return v;
    const auto outType = mlir::RankedTensorType::get(vType.getShape(), t);
    return mlir::tosa::CastOp::create(rewriter, loc, outType, v).getResult();
}

[[nodiscard]] inline mlir::Value ensureIntType(const mlir::Value currentVal, const mlir::Type targetElementType,
                                               mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                               const std::vector<float> &mins, const std::vector<float> &maxs,
                                               const bool symmetric, const std::string &partition_kind,
                                               const std::optional<int64_t> axis = std::nullopt) {
    const auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(currentVal.getType());
    if (!tensorType || tensorType.getElementType() == targetElementType) {
        return currentVal;
    }

    const mlir::Type sourceElementType = tensorType.getElementType();
    const mlir::Type targetTensorType = mlir::RankedTensorType::get(tensorType.getShape(), targetElementType);

    const bool isFloatToInt = !sourceElementType.isIntOrIndex() && targetElementType.isIntOrIndex();
    const bool isIntToFloat = sourceElementType.isIntOrIndex() && !targetElementType.isIntOrIndex();

    if (!isFloatToInt && !isIntToFloat) {
        auto castOp = mlir::tosa::CastOp::create(rewriter, loc, targetTensorType, currentVal);
        castOp->setAttr("conquer.cast", rewriter.getUnitAttr());
        return castOp.getResult();
    }

    const mlir::Type qParamElementType = isFloatToInt ? targetElementType : sourceElementType;
    std::vector<double> scales(mins.size());
    int64_t zp = 0;

    bool effectiveSymmetric = symmetric;
    if (qParamElementType.isIntOrIndex()) {
        effectiveSymmetric = getLegalTosaIntSymmetry(qParamElementType, symmetric);
        if (requiresSymmetricPartitionedIntQuant(partition_kind)) {
            effectiveSymmetric = true;
        }
    }

    for (size_t i = 0; i < mins.size(); ++i) {
        int64_t tempZp = 0;
        computeScaleAndZPForTosaInt(mins[i], maxs[i], qParamElementType, effectiveSymmetric, scales[i], tempZp);
        if (i == 0) zp = tempZp;
    }

    const llvm::StringRef opName = isFloatToInt ? "conquer.quantise" : "conquer.dequantise";
    mlir::OperationState state(loc, opName);
    state.addOperands(currentVal);
    state.addTypes(targetTensorType);
    state.addAttribute("scales", rewriter.getDenseF64ArrayAttr(scales));
    state.addAttribute("zero_point", rewriter.getI64IntegerAttr(zp));
    state.addAttribute("partition_kind", rewriter.getStringAttr(partition_kind));

    if (axis.has_value()) {
        state.addAttribute("axis", rewriter.getI64IntegerAttr(*axis));
    }
    state.addAttribute("narrow_range", rewriter.getBoolAttr(effectiveSymmetric));

    return rewriter.create(state)->getResult(0);
}

[[nodiscard]] inline mlir::Value applyMixedPrecisionIntCast(const mlir::Value operand,
                                                            const mlir::Type targetStorageType,
                                                            const mlir::Type computeType,
                                                            mlir::PatternRewriter &rewriter,
                                                            const mlir::Location loc,
                                                            const std::vector<float> &mins,
                                                            const std::vector<float> &maxs,
                                                            const bool symmetric,
                                                            const std::string &partition_kind,
                                                            const std::optional<int64_t> axis = std::nullopt) {
    mlir::Value quantised = ensureIntType(
        operand, targetStorageType, rewriter, loc, mins, maxs, symmetric, partition_kind, axis
    );

    if (targetStorageType != computeType) {
        const auto tensorType = llvm::cast<mlir::RankedTensorType>(quantised.getType());
        const auto targetTensorType = mlir::RankedTensorType::get(tensorType.getShape(), computeType);
        auto castOp = mlir::tosa::CastOp::create(rewriter, loc, targetTensorType, quantised);
        castOp->setAttr("conquer.cast", rewriter.getUnitAttr());
        quantised = castOp.getResult();
    }
    return quantised;
}

[[nodiscard]] inline std::pair<mlir::Value, int64_t> quantiseActivation(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, const mlir::Value rawInput,
    const mlir::Type targetStorageType, const mlir::Type computeType, const bool symmetric,
    const CalibrationMethod calibMethod) {

    const mlir::Value cleanInput = stripTransientCast(rawInput);
    mlir::Operation *defOp = cleanInput.getDefiningOp() ? cleanInput.getDefiningOp() : op;

    float aMin = 0.0f, aMax = 0.0f;
    if (!getActivationStats(defOp, calibMethod, targetStorageType.getIntOrFloatBitWidth(), aMin, aMax)) {
        op->emitWarning("Missing activation stats. Cannot quantise.");
        return {nullptr, 0};
    }

    const bool effectiveSymmetric = getLegalTosaIntSymmetry(targetStorageType, symmetric);

    double unusedScale = 1.0;
    int64_t zpVal = 0;
    computeScaleAndZPForTosaInt(aMin, aMax, targetStorageType, effectiveSymmetric, unusedScale, zpVal);

    const mlir::Value qVal = applyMixedPrecisionIntCast(
        cleanInput, targetStorageType, computeType, rewriter, op->getLoc(),
        {aMin}, {aMax}, effectiveSymmetric, "per_tensor"
    );
    return {qVal, zpVal};
}

[[nodiscard]] inline std::pair<mlir::Value, int64_t> quantiseWeight(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, const unsigned operandIdx,
    const mlir::Value rawWeight, const mlir::Type targetStorageType, const mlir::Type computeType,
    const bool symmetric, const CalibrationMethod calibMethod, const Granularity granularity) {

    const mlir::Value cleanWeight = stripTransientCast(rawWeight);

    std::vector<float> mins, maxs;
    std::string pk = "per_tensor";
    std::optional<int64_t> axis = std::nullopt;
    const unsigned bw = targetStorageType.getIntOrFloatBitWidth();

    bool isPerChannel = std::holds_alternative<PerChannelGranularity>(granularity);

    if (isPerChannel) {
        if (!getWeightStats(op, operandIdx, calibMethod, GranularityType::PerChannel, bw, mins, maxs, pk, axis)) {
            getWeightStats(op, operandIdx, calibMethod, GranularityType::PerTensor, bw, mins, maxs, pk, axis);
            isPerChannel = false;
        }
    } else {
        getWeightStats(op, operandIdx, calibMethod, GranularityType::PerTensor, bw, mins, maxs, pk, axis);
    }

    if (mins.empty() || maxs.empty()) {
        op->emitWarning("Missing weight stats. Cannot quantise.");
        return {nullptr, 0};
    }

    // Partitioned integer weight quantisation must be symmetric because the
    // quant op only stores a scalar zero-point.
    const bool effectiveSymmetric =
        (isPerChannel || requiresSymmetricPartitionedIntQuant(pk))
            ? true
            : getLegalTosaIntSymmetry(targetStorageType, symmetric);

    int64_t zpVal = 0;
    if (!(isPerChannel || requiresSymmetricPartitionedIntQuant(pk))) {
        double unusedScale = 1.0;
        computeScaleAndZPForTosaInt(mins[0], maxs[0], targetStorageType, effectiveSymmetric, unusedScale, zpVal);
    }

    const mlir::Value qVal = applyMixedPrecisionIntCast(
        cleanWeight, targetStorageType, computeType, rewriter, op->getLoc(),
        mins, maxs, effectiveSymmetric, pk, axis
    );
    return {qVal, zpVal};
}

[[nodiscard]] inline mlir::Value quantiseBias(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                              const mlir::Value bias, const mlir::Type targetType,
                                              const std::vector<double> &scales) {
    const auto biasType = llvm::cast<mlir::RankedTensorType>(bias.getType());
    const auto newBiasType = mlir::RankedTensorType::get(biasType.getShape(), targetType);

    const std::string pk = scales.size() > 1 ? "axis" : "per_tensor";
    const std::optional<int64_t> axis = scales.size() > 1 ? std::optional<int64_t>(0) : std::nullopt;

    mlir::OperationState state(loc, "conquer.quantise");
    state.addOperands(bias);
    state.addTypes(newBiasType);
    state.addAttribute("scales", rewriter.getDenseF64ArrayAttr(scales));
    state.addAttribute("zero_point", rewriter.getI64IntegerAttr(0));
    state.addAttribute("partition_kind", rewriter.getStringAttr(pk));

    if (axis.has_value()) {
        state.addAttribute("axis", rewriter.getI64IntegerAttr(*axis));
    }
    state.addAttribute("narrow_range", rewriter.getBoolAttr(false));

    return rewriter.create(state)->getResult(0);
}

[[nodiscard]] inline mlir::Value emitRescaleAndDequantise(
    mlir::PatternRewriter &rewriter, mlir::Operation *originalOp, const mlir::Value accumulator,
    const mlir::Type targetActType, const std::vector<double> &scaleA, const std::vector<double> &scaleB,
    const bool outSymmetric, const CalibrationMethod actCalib) {

    const mlir::Location loc = originalOp->getLoc();
    if (scaleA.empty() || scaleB.empty()) {
        originalOp->emitWarning("Missing operand scales. Cannot emit integer rescale.");
        return {};
    }

    float outMin = 0.0f, outMax = 0.0f;
    if (!getActivationStats(originalOp, actCalib, targetActType.getIntOrFloatBitWidth(), outMin, outMax)) {
        originalOp->emitWarning("Missing activation stats. Cannot emit integer rescale.");
        return {};
    }

    const bool effectiveOutSymmetric = getLegalTosaIntSymmetry(targetActType, outSymmetric);

    double outScale = 1.0;
    int64_t outZp = 0;
    computeScaleAndZPForTosaInt(outMin, outMax, targetActType, effectiveOutSymmetric, outScale, outZp);

    const size_t channels = std::max(scaleA.size(), scaleB.size());
    const auto accRanked = llvm::cast<mlir::RankedTensorType>(accumulator.getType());
    const auto accElemType = accRanked.getElementType();

    mlir::Value multConst;
    mlir::Value shiftConst;
    mlir::BoolAttr scale32Attr;

    if (accElemType.isInteger(48)) {
        std::vector<int16_t> multipliers(channels);
        std::vector<int8_t> shifts(channels);

        for (size_t i = 0; i < channels; ++i) {
            const double sA = scaleA.size() > 1 ? scaleA[i] : scaleA[0];
            const double sB = scaleB.size() > 1 ? scaleB[i] : scaleB[0];

            int16_t m = 0;
            int8_t s = 0;
            if (!tryComputeMultiplierAndShift16((sA * sB) / outScale, m, s)) {
                originalOp->emitWarning("Unable to represent i48 rescale as legal TOSA i16 multiplier / shift.");
                return {};
            }

            multipliers[i] = m;
            shifts[i] = s;
        }

        const auto multType = mlir::RankedTensorType::get({static_cast<int64_t>(channels)}, rewriter.getI16Type());
        const auto shiftType = mlir::RankedTensorType::get({static_cast<int64_t>(channels)}, rewriter.getI8Type());

        multConst = mlir::tosa::ConstOp::create(
            rewriter, loc, multType,
            mlir::DenseElementsAttr::get(multType, llvm::ArrayRef<int16_t>(multipliers))
        ).getResult();

        shiftConst = mlir::tosa::ConstOp::create(
            rewriter, loc, shiftType,
            mlir::DenseElementsAttr::get(shiftType, llvm::ArrayRef<int8_t>(shifts))
        ).getResult();

        scale32Attr = rewriter.getBoolAttr(false);
    } else {
        std::vector<int32_t> multipliers(channels);
        std::vector<int8_t> shifts(channels);

        for (size_t i = 0; i < channels; ++i) {
            const double sA = scaleA.size() > 1 ? scaleA[i] : scaleA[0];
            const double sB = scaleB.size() > 1 ? scaleB[i] : scaleB[0];

            int32_t m = 0;
            int8_t s = 0;
            if (!tryComputeMultiplierAndShift32((sA * sB) / outScale, m, s)) {
                originalOp->emitWarning("Unable to represent i32 rescale as legal TOSA i32 multiplier / shift.");
                return {};
            }

            multipliers[i] = m;
            shifts[i] = s;
        }

        const auto multType = mlir::RankedTensorType::get({static_cast<int64_t>(channels)}, rewriter.getI32Type());
        const auto shiftType = mlir::RankedTensorType::get({static_cast<int64_t>(channels)}, rewriter.getI8Type());

        multConst = mlir::tosa::ConstOp::create(
            rewriter, loc, multType,
            mlir::DenseElementsAttr::get(multType, llvm::ArrayRef<int32_t>(multipliers))
        ).getResult();

        shiftConst = mlir::tosa::ConstOp::create(
            rewriter, loc, shiftType,
            mlir::DenseElementsAttr::get(shiftType, llvm::ArrayRef<int8_t>(shifts))
        ).getResult();

        scale32Attr = rewriter.getBoolAttr(true);
    }

    const auto inZpType = mlir::RankedTensorType::get({1}, accElemType);
    const auto inZpConst = mlir::tosa::ConstOp::create(
        rewriter, loc, inZpType, llvm::cast<mlir::DenseElementsAttr>(rewriter.getZeroAttr(inZpType))
    ).getResult();

    const auto outZpType = mlir::RankedTensorType::get({1}, targetActType);
    const auto outZpElemAttr = rewriter.getIntegerAttr(targetActType, outZp);
    const auto outZpConst = mlir::tosa::ConstOp::create(
        rewriter, loc, outZpType, mlir::DenseElementsAttr::get(outZpType, outZpElemAttr)
    ).getResult();

    const auto rescaledType = mlir::RankedTensorType::get(accRanked.getShape(), targetActType);

    auto rescaleOp = mlir::tosa::RescaleOp::create(
        rewriter, loc, rescaledType, accumulator,
        multConst, shiftConst, inZpConst, outZpConst,
        scale32Attr,
        mlir::tosa::RoundingModeAttr::get(rewriter.getContext(), mlir::tosa::RoundingMode::SINGLE_ROUND),
        rewriter.getBoolAttr(channels > 1),
        rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false)
    );

    const auto originalResultType = llvm::cast<mlir::RankedTensorType>(originalOp->getResult(0).getType());
    mlir::OperationState dequantState(loc, "conquer.dequantise");
    dequantState.addOperands(rescaleOp.getOutput());
    dequantState.addTypes(originalResultType);
    dequantState.addAttribute("scales", rewriter.getDenseF64ArrayAttr({outScale}));
    dequantState.addAttribute("zero_point", rewriter.getI64IntegerAttr(outZp));
    dequantState.addAttribute("partition_kind", rewriter.getStringAttr("per_tensor"));
    dequantState.addAttribute("narrow_range", rewriter.getBoolAttr(effectiveOutSymmetric));

    return rewriter.create(dequantState)->getResult(0);
}

struct QuantParams {
    mlir::Type targetActType;
    mlir::Type targetWgtType;
    bool actIsInt{};
    bool wgtIsInt{};
    bool actSym{};
    bool wgtSym{};
    CalibrationMethod actCalib;
    CalibrationMethod wgtCalib;
    Granularity wgtGran;
};

[[nodiscard]] inline std::optional<QuantParams> extractQuantParams(mlir::Operation *op,
                                                                   const mlir::PatternRewriter &rewriter,
                                                                   const QuantisationPolicy &policy) {
    const auto [tgt_wgt, tgt_act] = get_target_type(*rewriter.getContext(), op, policy);
    if (!tgt_wgt.has_value() && !tgt_act.has_value()) return std::nullopt;

    QuantParams p;
    p.targetWgtType = tgt_wgt.value_or(mlir::Float32Type::get(rewriter.getContext()));
    p.targetActType = tgt_act.value_or(mlir::Float32Type::get(rewriter.getContext()));
    p.wgtIsInt = p.targetWgtType.isIntOrIndex();
    p.actIsInt = p.targetActType.isIntOrIndex();

    if (!p.wgtIsInt && !p.actIsInt) return std::nullopt;

    // Keep this policy-pure. TOSA legality repair is applied later by the
    // integer quantisation helpers.
    p.actSym = is_activation_symmetric(op, policy);
    p.wgtSym = is_weight_symmetric(op, policy);
    p.actCalib = get_activation_calibration_method(op, policy);
    p.wgtCalib = get_weight_calibration_method(op, policy);
    p.wgtGran = get_weight_granularity(op, policy);

    return p;
}

// Helper for scalar / per-tensor scale32=true TOSA rescale creation.
// This remains the 32-bit multiplier path; i48 accumulation should use
// emitRescaleAndDequantise or another dedicated scale32=false helper.
[[nodiscard]] inline mlir::Value createTosaRescale(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                                   mlir::Value input, int32_t multiplier, int8_t shift,
                                                   int64_t in_zp, int64_t out_zp, mlir::Type out_type) {
    const auto inElemType = llvm::cast<mlir::RankedTensorType>(input.getType()).getElementType();

    const int64_t legalInZp = getLegalTosaIntZeroPoint(inElemType, in_zp);
    const int64_t legalOutZp = getLegalTosaIntZeroPoint(out_type, out_zp);

    const auto multType = mlir::RankedTensorType::get({1}, rewriter.getI32Type());
    const auto shiftType = mlir::RankedTensorType::get({1}, rewriter.getI8Type());
    const auto inZpType = mlir::RankedTensorType::get({1}, inElemType);
    const auto outZpType = mlir::RankedTensorType::get({1}, out_type);

    const auto multConst = mlir::tosa::ConstOp::create(
        rewriter, loc, multType,
        mlir::DenseElementsAttr::get(multType, llvm::ArrayRef<int32_t>({multiplier}))
    ).getResult();

    const auto shiftConst = mlir::tosa::ConstOp::create(
        rewriter, loc, shiftType,
        mlir::DenseElementsAttr::get(shiftType, llvm::ArrayRef<int8_t>({shift}))
    ).getResult();

    const auto inZpAttr = rewriter.getIntegerAttr(inElemType, legalInZp);
    const auto inZpConst = mlir::tosa::ConstOp::create(
        rewriter, loc, inZpType, mlir::DenseElementsAttr::get(inZpType, inZpAttr)
    ).getResult();

    const auto outZpAttr = rewriter.getIntegerAttr(out_type, legalOutZp);
    const auto outZpConst = mlir::tosa::ConstOp::create(
        rewriter, loc, outZpType, mlir::DenseElementsAttr::get(outZpType, outZpAttr)
    ).getResult();

    const auto outShape = llvm::cast<mlir::RankedTensorType>(input.getType()).getShape();
    const auto rescaledType = mlir::RankedTensorType::get(outShape, out_type);

    auto rescaleOp = mlir::tosa::RescaleOp::create(
        rewriter, loc, rescaledType, input,
        multConst, shiftConst, inZpConst, outZpConst,
        rewriter.getBoolAttr(true), // scale32
        mlir::tosa::RoundingModeAttr::get(rewriter.getContext(), mlir::tosa::RoundingMode::SINGLE_ROUND),
        rewriter.getBoolAttr(false), // per_channel
        rewriter.getBoolAttr(false), // input_unsigned
        rewriter.getBoolAttr(false)  // output_unsigned
    );

    return rescaleOp.getOutput();
}

[[nodiscard]] inline llvm::SmallVector<int64_t> getSameRankBroadcastShape(const mlir::RankedTensorType tensorType) {
    return llvm::SmallVector<int64_t>(tensorType.getRank(), 1);
}

[[nodiscard]] inline bool preflightCheckQuantStats(mlir::Operation *op, const QuantParams &p) {
    if (p.actIsInt) {
        float dummyMin = 0.0f, dummyMax = 0.0f;
        if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), dummyMin, dummyMax)) {
            return false;
        }
    }

    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        if (i >= 2) continue;

        mlir::Value operand = op->getOperand(i);

        if (is_weight(operand)) {
            if (p.wgtIsInt) {
                std::vector<float> mins, maxs;
                std::string pk;
                std::optional<int64_t> axis;
                const unsigned bw = p.targetWgtType.getIntOrFloatBitWidth();
                const auto reqGran = std::holds_alternative<PerChannelGranularity>(p.wgtGran)
                    ? GranularityType::PerChannel : GranularityType::PerTensor;

                if (!getWeightStats(op, i, p.wgtCalib, reqGran, bw, mins, maxs, pk, axis) &&
                    !getWeightStats(op, i, p.wgtCalib, GranularityType::PerTensor, bw, mins, maxs, pk, axis)) {
                    return false;
                    }
            }
        } else {
            if (p.actIsInt) {
                const mlir::Value cleanInput = stripTransientCast(operand);
                mlir::Operation *defOp = cleanInput.getDefiningOp() ? cleanInput.getDefiningOp() : op;

                float inMin = 0.0f, inMax = 0.0f;
                if (!getActivationStats(defOp, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), inMin, inMax)) {
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace conquer
