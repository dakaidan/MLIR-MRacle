#pragma once

#include "conquer/core/logging.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace conquer::integer_quant {

namespace detail {

[[nodiscard]] inline int64_t saturatingRoundToI64(const double value) {
    if (std::isnan(value)) return 0;
    if (value <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    if (value >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(std::llround(value));
}

[[nodiscard]] inline int64_t saturatingAddI64(const int64_t a, const int64_t b) {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
        return std::numeric_limits<int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
        return std::numeric_limits<int64_t>::min();
    }
    return a + b;
}

} // namespace detail

template <typename OpTy>
struct QuantiseScalePreservingPattern : public mlir::RewritePattern {
    QuantiseScalePreservingPattern(mlir::MLIRContext *context)
        : RewritePattern(OpTy::getOperationName(), 1, context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        L_TRACE("Applying QuantiseScalePreservingPattern to: " << compact(op));
        mlir::Operation *quantOp = nullptr;
        for (auto *user : op->getUsers()) {
            if (user->getName().getStringRef() == "conquer.quantise") {
                quantOp = user;
                break;
            }
        }

        if (!quantOp) return mlir::failure();

        auto outType = llvm::cast<mlir::RankedTensorType>(quantOp->getResult(0).getType()).getElementType();
        if (!outType.isIntOrIndex()) return mlir::failure();

        const unsigned bitWidth = outType.getIntOrFloatBitWidth();
        if (bitWidth != 8 && bitWidth != 16) return mlir::failure();

        const mlir::Location loc = op->getLoc();
        llvm::SmallVector<mlir::Value> newOperands;

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            mlir::Value operand = op->getOperand(i);
            auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());

            if (rankedType && !rankedType.getElementType().isIntOrIndex()) {
                mlir::OperationState qState(loc, "conquer.quantise");
                qState.addOperands(operand);
                qState.addTypes(mlir::RankedTensorType::get(rankedType.getShape(), outType));
                qState.addAttributes(quantOp->getAttrs());

                auto newQuant = rewriter.create(qState);
                newOperands.push_back(newQuant->getResult(0));
            } else {
                newOperands.push_back(operand);
            }
        }

        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), outType);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue() == "quantization_info") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());
        auto newOp = rewriter.create(state);

        rewriter.replaceOp(quantOp, newOp->getResult(0));

        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }

        return mlir::success();
    }
};

struct QuantiseClampPattern : public mlir::OpRewritePattern<mlir::tosa::ClampOp> {
    QuantiseClampPattern(mlir::MLIRContext *context)
        : OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::ClampOp op, mlir::PatternRewriter &rewriter) const override {
        L_TRACE("Applying QuantiseClampPattern to: " << compact(op));
        mlir::Operation *quantOp = nullptr;
        for (auto *user : op->getUsers()) {
            if (user->getName().getStringRef() == "conquer.quantise") {
                quantOp = user;
                break;
            }
        }
        if (!quantOp) return mlir::failure();

        auto outType = llvm::cast<mlir::RankedTensorType>(quantOp->getResult(0).getType()).getElementType();
        if (!outType.isIntOrIndex()) return mlir::failure();

        const unsigned bitWidth = outType.getIntOrFloatBitWidth();
        if (bitWidth != 8 && bitWidth != 16) return mlir::failure();

        const mlir::Location loc = op->getLoc();

        auto scalesAttr = quantOp->getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
        if (!scalesAttr || scalesAttr.asArrayRef().size() != 1) return mlir::failure();

        auto zpAttr = quantOp->getAttrOfType<mlir::IntegerAttr>("zero_point");
        auto narrowAttr = quantOp->getAttrOfType<mlir::BoolAttr>("narrow_range");

        double scale = scalesAttr.asArrayRef()[0];
        int64_t zp = zpAttr ? zpAttr.getInt() : 0;
        bool symmetric = narrowAttr ? narrowAttr.getValue() : false;

        if (!std::isfinite(scale) || scale <= 0.0) return mlir::failure();

        auto rankedType = llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
        mlir::OperationState qState(loc, "conquer.quantise");
        qState.addOperands(op.getInput());
        qState.addTypes(mlir::RankedTensorType::get(rankedType.getShape(), outType));
        qState.addAttributes(quantOp->getAttrs());
        auto newQuant = rewriter.create(qState);

        auto minAttr = op.getMinValAttr();
        auto maxAttr = op.getMaxValAttr();

        if (!llvm::isa<mlir::FloatAttr>(minAttr) || !llvm::isa<mlir::FloatAttr>(maxAttr)) {
            return mlir::failure();
        }

        const double min_f = llvm::cast<mlir::FloatAttr>(minAttr).getValueAsDouble();
        const double max_f = llvm::cast<mlir::FloatAttr>(maxAttr).getValueAsDouble();

        if (!std::isfinite(min_f) && !std::isinf(min_f)) return mlir::failure();
        if (!std::isfinite(max_f) && !std::isinf(max_f)) return mlir::failure();
        if (min_f > max_f) return mlir::failure();

        auto quantiseBound = [&](const double value) -> int64_t {
            const int64_t rounded = detail::saturatingRoundToI64(value / scale);
            return detail::saturatingAddI64(rounded, zp);
        };

        int64_t q_min = quantiseBound(min_f);
        int64_t q_max = quantiseBound(max_f);

        const int64_t type_min = symmetric ? -(1LL << (bitWidth - 1)) + 1 : -(1LL << (bitWidth - 1));
        const int64_t type_max = (1LL << (bitWidth - 1)) - 1;

        q_min = std::clamp(q_min, type_min, type_max);
        q_max = std::clamp(q_max, type_min, type_max);

        // This should not happen for valid float clamp ranges with positive scale,
        // but guard anyway against bad metadata / numerical pathologies.
        if (q_min > q_max) return mlir::failure();

        const auto origResultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
        const auto newResultType = mlir::RankedTensorType::get(origResultType.getShape(), outType);

        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newQuant->getResult(0));
        state.addTypes(newResultType);

        for (auto attr : op->getAttrs()) {
            if (attr.getName() == "min_val" || attr.getName() == "max_val" || attr.getName() == "quantization_info") {
                continue;
            }
            state.addAttribute(attr.getName(), attr.getValue());
        }

        state.addAttribute("min_val", rewriter.getIntegerAttr(outType, q_min));
        state.addAttribute("max_val", rewriter.getIntegerAttr(outType, q_max));
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        auto newOp = rewriter.create(state);

        rewriter.replaceOp(quantOp, newOp->getResult(0));

        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }

        return mlir::success();
    }
};

inline void populateLayoutPatterns(mlir::RewritePatternSet &patterns) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ReshapeOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::TransposeOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ReverseOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::SliceOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::TileOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::PadOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::GatherOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ScatterOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ConcatOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::MaxPool2dOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ReduceMaxOp>>(context);
    patterns.add<QuantiseScalePreservingPattern<mlir::tosa::ReduceMinOp>>(context);
    patterns.add<QuantiseClampPattern>(context);
}

} // namespace conquer::integer_quant
