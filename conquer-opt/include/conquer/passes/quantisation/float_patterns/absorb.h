#pragma once

#include "conquer/passes/quantisation/float_patterns/utilities.h"
#include "conquer/passes/quantisation/shared/quant_plan.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <algorithm>

namespace conquer::float_quant {

// Generic Hoister for ops that don't change data in a way that requires
// a dedicated accumulator plan.
// By pulling casts through these ops, they natively absorb the precision of
// their float inputs, but only at TOSA-legal types.
template <typename OpTy>
struct AbsorbPrecisionPattern : mlir::OpRewritePattern<mlir::tosa::CastOp> {
    using mlir::OpRewritePattern<mlir::tosa::CastOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::tosa::CastOp castOp,
                                        mlir::PatternRewriter &rewriter) const override {
        if (!castOp->hasAttr("conquer.cast")) return mlir::failure();

        auto op = castOp.getInput().getDefiningOp<OpTy>();
        if (!op) return mlir::failure();

        L_TRACE("Applying AbsorbPrecisionPattern to: " << compact(op));

        // If theres more than one user, we only quantise if they are all casts to the same type
        if (!op->getResult(0).hasOneUse()) {
            const auto other_users = op->getResult(0).getUsers();
            for (auto user : other_users) {
                if (user == castOp) continue;
                auto other_cast = llvm::dyn_cast<mlir::tosa::CastOp>(user);
                if (!other_cast) return mlir::failure();
                if (other_cast.getType() != castOp.getType()) return mlir::failure();
            }

        }

        const auto finalType = llvm::dyn_cast<mlir::RankedTensorType>(castOp.getType());
        if (!finalType) return mlir::failure();

        const mlir::Type requestedFinalElemType = finalType.getElementType();

        if (!mlir::isa<mlir::FloatType>(requestedFinalElemType)) return mlir::failure();

        const auto absorbPlan =
            planTosaAbsorbFloat<OpTy>(rewriter.getContext(), requestedFinalElemType);
        if (!absorbPlan) return mlir::failure();

        const mlir::Type opElemType = absorbPlan->opType;
        const mlir::Type finalElemType = absorbPlan->finalType;

        const auto currentResultType =
            llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!currentResultType) return mlir::failure();

        // If the op is already at the best legal absorbed type, then:
        // - if final type matches too, there is nothing to do
        // - if final type differs, we must keep the trailing cast anyway, so no absorb benefit
        if (currentResultType.getElementType() == opElemType) {
            return mlir::failure();
        }

        // Hoist casts ABOVE the operation for all float tensor operands.
        llvm::SmallVector<mlir::Value, 4> newOperands;
        for (mlir::Value operand : op->getOperands()) {
            auto tensorType = llvm::dyn_cast<mlir::TensorType>(operand.getType());

            // Some ops (like Select or Gather) take boolean/integer indices.
            // We only cast float data operands.
            if (tensorType && mlir::isa<mlir::FloatType>(tensorType.getElementType())) {
                newOperands.push_back(ensureType(operand, opElemType, rewriter, castOp.getLoc()));
            } else {
                newOperands.push_back(operand);
            }
        }

        const mlir::Type newOpType = mlir::RankedTensorType::get(
            currentResultType.getShape(), opElemType);

        mlir::OperationState state(castOp.getLoc(), op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newOpType);

        if (auto clampOp = llvm::dyn_cast<mlir::tosa::ClampOp>(op.getOperation())) {
            const auto &sem = llvm::cast<mlir::FloatType>(opElemType).getFloatSemantics();
            const double typeMax = llvm::APFloat::getLargest(sem, false).convertToDouble();
            const double typeMin = llvm::APFloat::getLargest(sem, true).convertToDouble();

            const double originalMin =
                mlir::cast<mlir::FloatAttr>(clampOp.getMinVal()).getValue().convertToDouble();
            const double originalMax =
                mlir::cast<mlir::FloatAttr>(clampOp.getMaxVal()).getValue().convertToDouble();

            state.addAttribute(
                "min_val",
                rewriter.getFloatAttr(opElemType, std::clamp(originalMin, typeMin, typeMax)));
            state.addAttribute(
                "max_val",
                rewriter.getFloatAttr(opElemType, std::clamp(originalMax, typeMin, typeMax)));
        } else {
            state.addAttributes(op->getAttrs());
        }

        mlir::Operation *newOp = rewriter.create(state);

        mlir::Value replacement = newOp->getResult(0);

        // If the op couldn't legally absorb all the way to the requested final type
        // (e.g. requested fp8 for exp/select/resize/etc), preserve the trailing cast.
        if (absorbPlan->requiresTrailingCast()) {
            replacement = ensureType(replacement, finalElemType, rewriter, castOp.getLoc());
        }

        rewriter.replaceOp(castOp, replacement);
        return mlir::success();
    }
};

inline void populateAbsorbPatterns(mlir::RewritePatternSet &patterns) {
    mlir::MLIRContext *context = patterns.getContext();

    // === Data Layout & Image ===
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ConcatOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::PadOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ReshapeOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ReverseOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::SliceOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::TileOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::TransposeOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ResizeOp>>(context);

    // === Scatter/Gather ===
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::GatherOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ScatterOp>>(context);

    // === Ternary ===
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::SelectOp>>(context);

    // === Routing Ops ===
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::MaxPool2dOp>>(context);

    // === Activations & Unaries ===
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ClampOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ErfOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::SigmoidOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::TanhOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::AbsOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::CeilOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::CosOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ExpOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::FloorOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::LogOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::NegateOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::ReciprocalOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::RsqrtOp>>(context);
    patterns.add<AbsorbPrecisionPattern<mlir::tosa::SinOp>>(context);
}

} // namespace conquer
