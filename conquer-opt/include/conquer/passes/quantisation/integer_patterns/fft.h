#pragma once

#include "conquer/passes/quantisation/integer_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace conquer::integer_quant {

template <typename OpTy>
struct QuantiseFFTPattern : public mlir::RewritePattern {
    QuantiseFFTPattern(mlir::MLIRContext *context, QuantisationPolicy policy)
        : RewritePattern(OpTy::getOperationName(), 1, context), quantPolicy(std::move(policy)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        L_TRACE("Applying QuantiseFFTPattern to: " << compact(op));
        if (op->hasAttr("conquer.int.quantised")) return mlir::failure();

        // FFTs have explicit configs, so we respect should_quantise
        if (!should_quantise(op, quantPolicy)) return mlir::failure();

        const auto p_opt = extractQuantParams(op, rewriter, quantPolicy);
        if (!p_opt.has_value()) return mlir::failure();
        const auto& p = p_opt.value();

        if (!preflightCheckQuantStats(op, p)) return mlir::failure();

        const mlir::Location loc = op->getLoc();

        // TOSA mandates f32 execution for FFTs
        const mlir::Type exec_type = mlir::Float32Type::get(rewriter.getContext());

        // 1. Prepare Inputs (Real and optionally Imaginary)
        llvm::SmallVector<mlir::Value> newOperands;
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            mlir::Value operand = op->getOperand(i);

            if (p.actIsInt) {
                // Quantise the input to the requested storage type
                auto [resValue, _] = quantiseActivation(rewriter, op, operand, p.targetActType, p.targetActType, p.actSym, p.actCalib);
                if (!resValue) return mlir::failure();

                float aMin = 0.0f, aMax = 0.0f;
                if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), aMin, aMax)) return mlir::failure();

                // Immediately push to f32 for execution
                auto finalVal = ensureIntType(resValue, exec_type, rewriter, loc, {aMin}, {aMax}, p.actSym, "per_tensor");
                newOperands.push_back(finalVal);
            } else {
                newOperands.push_back(castToType(rewriter, loc, stripTransientCast(operand), exec_type));
            }
        }

        // 2. Prepare Outputs (Real and Imaginary results)
        llvm::SmallVector<mlir::Type> newResultTypes;
        for (auto res : op->getResults()) {
            auto origShape = llvm::cast<mlir::RankedTensorType>(res.getType()).getShape();
            newResultTypes.push_back(mlir::RankedTensorType::get(origShape, exec_type));
        }

        // 3. Emit the f32 FFT Operation
        mlir::OperationState state(loc, op->getName().getStringRef());
        state.addOperands(newOperands);
        state.addTypes(newResultTypes);

        for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue() == "quantization_info") continue;
            state.addAttribute(attr.getName(), attr.getValue());
        }
        state.addAttribute("conquer.int.quantised", rewriter.getUnitAttr());

        auto newOp = rewriter.create(state);

        // 4. Quantise all outputs back to the requested format
        llvm::SmallVector<mlir::Value> finalOuts;
        for (unsigned i = 0; i < newOp->getNumResults(); ++i) {
            mlir::Value out = newOp->getResult(i);
            auto origType = llvm::cast<mlir::RankedTensorType>(op->getResult(i).getType()).getElementType();

            if (p.actIsInt) {
                float outMin = 0.0f, outMax = 0.0f;
                if (!getActivationStats(op, p.actCalib, p.targetActType.getIntOrFloatBitWidth(), outMin, outMax)) return mlir::failure();

                // Quantise down to requested storage type (e.g. i8)
                const mlir::Value qOut = ensureIntType(out, p.targetActType, rewriter, loc, {outMin}, {outMax}, p.actSym, "per_tensor");

                // Dequantise back up to original graph precision boundary
                finalOuts.push_back(ensureIntType(qOut, origType, rewriter, loc, {outMin}, {outMax}, p.actSym, "per_tensor"));
            } else {
                finalOuts.push_back(castToType(rewriter, loc, out, origType));
            }
        }

        rewriter.replaceOp(op, finalOuts);
        return mlir::success();
    }

private:
    QuantisationPolicy quantPolicy;
};

inline void populateFFTPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    mlir::MLIRContext *context = patterns.getContext();

    patterns.add<QuantiseFFTPattern<mlir::tosa::FFT2dOp>>(context, policy);
    patterns.add<QuantiseFFTPattern<mlir::tosa::RFFT2dOp>>(context, policy);
}

}