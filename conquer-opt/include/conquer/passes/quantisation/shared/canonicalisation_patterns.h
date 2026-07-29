#pragma once

#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/Support/Debug.h>

namespace conquer {
// Handles: cast, const

struct QuantiseCastPattern : mlir::OpRewritePattern<mlir::tosa::CastOp> {
    QuantiseCastPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::CastOp op, mlir::PatternRewriter &rewriter) const override {
        if (!op->hasAttr("conquer.cast") || op->hasAttr("conquer.int.transient")) return mlir::failure();

        auto prevCast = op.getInput().getDefiningOp<mlir::tosa::CastOp>();
        if (!prevCast) return mlir::failure();

        const mlir::Value originalInput = prevCast.getInput();

        const mlir::Type typeA = llvm::cast<mlir::RankedTensorType>(originalInput.getType()).getElementType();
        const mlir::Type typeB = llvm::cast<mlir::RankedTensorType>(prevCast.getType()).getElementType();
        const mlir::Type typeC = llvm::cast<mlir::RankedTensorType>(op.getType()).getElementType();

        // 1. Redundant round-trip: cast(f16 -> f32 -> f16) == Identity
        if (typeA == typeC) {
            L_TRACE("Collapsing redundant round-trip cast.");
            rewriter.replaceOp(op, originalInput);
            return mlir::success();
        }

        // 2. Chained casts: cast(A -> B -> C)
        // Do NOT fold if B is smaller than C (e.g., f32 -> fp8 -> fp16).
        // If we fold that to (f32 -> fp16), we lose the fp8 storage compression.
        if (get_precision_rank(typeB) < get_precision_rank(typeA)) {
            return mlir::failure();
        }

        const bool is_f16_bf16_pair_AC = (typeA.isF16() && typeC.isBF16()) || (typeA.isBF16() && typeC.isF16());
        const bool is_f16_bf16_pair_AB = (typeA.isF16() && typeB.isBF16()) || (typeA.isBF16() && typeB.isF16());
        const bool is_f16_bf16_pair_BC = (typeB.isF16() && typeC.isBF16()) || (typeB.isBF16() && typeC.isF16());
        if (is_f16_bf16_pair_AC || is_f16_bf16_pair_AB || is_f16_bf16_pair_BC) {
            return mlir::failure();
        }

        L_TRACE("Collapsing chained casts into a single cast.");
        auto newCast = mlir::tosa::CastOp::create(rewriter, op.getLoc(), op.getType(), originalInput);
        newCast->setAttr("conquer.cast", rewriter.getUnitAttr());
        rewriter.replaceOp(op, newCast.getResult());

        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

struct QuantiseConstPattern : mlir::OpRewritePattern<mlir::tosa::ConstOp> {
    QuantiseConstPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : OpRewritePattern(context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::tosa::ConstOp op, mlir::PatternRewriter &rewriter) const override {
        llvm::SmallVector<mlir::tosa::CastOp, 4> castUsers;
        for (const auto user : op->getUsers()) {
            if (const auto castUser = llvm::dyn_cast<mlir::tosa::CastOp>(user)) {
                if (castUser->hasAttr("conquer.int.transient")) {
                    return mlir::failure();
                }
                castUsers.push_back(castUser);
            }
        }

        if (castUsers.empty()) return mlir::failure();

        mlir::Type highestTargetType = castUsers[0].getType();
        mlir::Type highestElemType = llvm::cast<mlir::RankedTensorType>(highestTargetType).getElementType();

        for (size_t i = 1; i < castUsers.size(); ++i) {
            mlir::Type targetType = castUsers[i].getType();
            const mlir::Type targetElemType = llvm::cast<mlir::RankedTensorType>(targetType).getElementType();

            if (get_precision_rank(targetElemType) > get_precision_rank(highestElemType)) {
                highestElemType = targetElemType;
                highestTargetType = targetType;
            }
        }

        if (highestTargetType == op.getType()) return mlir::failure();

        const mlir::Type sourceElemType = llvm::cast<mlir::RankedTensorType>(op.getType()).getElementType();

        // 1. Refuse to fold upcasts into constants (preserves lowest possible storage)
        if (get_precision_rank(highestElemType) > get_precision_rank(sourceElemType)) {
            return mlir::failure();
        }

        // 2. Only operate on floats
        if (!mlir::isa<mlir::FloatType>(highestElemType)) {
            return mlir::failure();
        }

        auto attr = op.getValues();
        const auto shapedType = llvm::cast<mlir::ShapedType>(op.getType());
        const size_t numElements = shapedType.getNumElements();

        llvm::SmallVector<float> extractedFloats;
        extractedFloats.reserve(numElements);

        if (const auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
            if (!mlir::isa<mlir::FloatType>(denseAttr.getElementType())) return mlir::failure();
            for (const float val : denseAttr.getValues<float>()) extractedFloats.push_back(val);
        } else if (const auto resourceAttr = llvm::dyn_cast<mlir::DenseResourceElementsAttr>(attr)) {
            const auto elemType = llvm::dyn_cast<mlir::FloatType>(resourceAttr.getElementType());
            if (!elemType) return mlir::failure();

            if (const mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob()) {
                const llvm::ArrayRef<char> rawBytes = blob->getData();

                // 1. f32
                if (elemType.isF32()) {
                    const auto f32Data = reinterpret_cast<const float *>(rawBytes.data());
                    extractedFloats.assign(f32Data, f32Data + numElements);
                }
                // 2. 16-bit (fp16, bf16)
                else if (elemType.getIntOrFloatBitWidth() == 16) {
                    const auto data = reinterpret_cast<const uint16_t*>(rawBytes.data());
                    const auto& sem = elemType.getFloatSemantics();
                    for (size_t i = 0; i < numElements; ++i) {
                        extractedFloats.push_back(llvm::APFloat(sem, llvm::APInt(16, data[i])).convertToFloat());
                    }
                }
                // 3. 8-bit (fp8e5m2, fp8e4m3fn)
                else if (elemType.getIntOrFloatBitWidth() == 8) {
                    const auto data = reinterpret_cast<const uint8_t*>(rawBytes.data());
                    const auto& sem = elemType.getFloatSemantics();
                    for (size_t i = 0; i < numElements; ++i) {
                        extractedFloats.push_back(llvm::APFloat(sem, llvm::APInt(8, data[i])).convertToFloat());
                    }
                } else {
                    return mlir::failure(); // Unknown width
                }
            } else {
                L_DEBUG("Failed to access underlying blob for resource.");
                return mlir::failure();
            }
        } else {
            return mlir::failure();
        }

        llvm::SmallVector<llvm::APFloat> newFloatValues;
        newFloatValues.reserve(numElements);

        const llvm::fltSemantics &targetSemantics = mlir::cast<mlir::FloatType>(highestElemType).getFloatSemantics();

        for (const float f : extractedFloats) {
            llvm::APFloat apf(f);
            bool losesInfo;
            apf.convert(targetSemantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            newFloatValues.push_back(apf);
        }

        const auto newRankedType = mlir::RankedTensorType::get(shapedType.getShape(), highestElemType);
        const auto newElementsAttr = mlir::DenseElementsAttr::get(newRankedType, newFloatValues);

        auto newConst = mlir::tosa::ConstOp::create(rewriter, op.getLoc(), highestTargetType, newElementsAttr);

        for (const mlir::NamedAttribute &attr : op->getAttrs()) {
            if (!attr.getName().strref().starts_with("value")) {
                newConst->setAttr(attr.getName(), attr.getValue());
            }
        }

        for (auto castOp : castUsers) {
            if (castOp.getType() == highestTargetType) {
                rewriter.replaceOp(castOp, newConst.getResult());
            } else {
                rewriter.modifyOpInPlace(castOp, [&]() { castOp.setOperand(newConst.getResult()); });
            }
        }

        rewriter.replaceOp(op, newConst.getResult());
        return mlir::success();
    }

  private:
    QuantisationPolicy quantPolicy;
};

inline void populateCanonicalisePatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseConstPattern>(context, policy);
    patterns.add<QuantiseCastPattern>(context, policy);
}
} // namespace conquer
