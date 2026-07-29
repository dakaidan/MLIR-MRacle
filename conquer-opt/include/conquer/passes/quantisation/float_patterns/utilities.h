#pragma once

#include "conquer/passes/quantisation/shared/utilities.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer {

inline mlir::Value ensureType(const mlir::Value currentVal, const mlir::Type targetElementType,
                              mlir::PatternRewriter &rewriter, const mlir::Location loc,
                              const bool mark_as_int = false) {
    const auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(currentVal.getType());
    if (!tensorType)
        return currentVal;

    const mlir::Type targetTensorType = mlir::RankedTensorType::get(tensorType.getShape(), targetElementType);

    if (currentVal.getType() == targetTensorType) {
        return currentVal;
    }

    mlir::tosa::CastOp castOp;

    const mlir::Type currentElemType = tensorType.getElementType();
    const bool current_type_is_16bit_float = currentElemType.isBF16() || currentElemType.isF16();
    const bool target_type_is_16bit_float = targetElementType.isBF16() || targetElementType.isF16();

    if (current_type_is_16bit_float && target_type_is_16bit_float && currentElemType != targetElementType) {
        // Bridge cast through F32
        auto bridge_cast = mlir::tosa::CastOp::create(rewriter, loc,
                                                        mlir::RankedTensorType::get(tensorType.getShape(),
                                                            static_cast<mlir::Type>(rewriter.getF32Type())),
                                                        currentVal);
        bridge_cast->setAttr("conquer.bridge", rewriter.getUnitAttr());
        castOp = mlir::tosa::CastOp::create(rewriter, loc, targetTensorType, bridge_cast.getResult());
    } else {
        castOp = mlir::tosa::CastOp::create(rewriter, loc, targetTensorType, currentVal);
    }

    castOp->setAttr("conquer.cast", rewriter.getUnitAttr());
    if (mark_as_int) {
        castOp->setAttr("conquer.int.transient", rewriter.getUnitAttr());
    }
    return castOp.getResult();
}

inline mlir::Value applyMixedPrecisionCast(
        const mlir::Value operand,
        const std::optional<mlir::Type> targetStorageType,
        const mlir::Type computeType,
        mlir::PatternRewriter &rewriter,
        const mlir::Location loc) {

    if (!targetStorageType.has_value()) {
        return ensureType(operand, computeType, rewriter, loc);
    }

    if (targetStorageType.value().isIntOrIndex()) {
        return ensureType(operand, computeType, rewriter, loc, true);
    }

    mlir::Value quantised = ensureType(operand, targetStorageType.value(), rewriter, loc);

    if (targetStorageType.value() != computeType) {
        quantised = ensureType(quantised, computeType, rewriter, loc);
    }

    return quantised;
}
}