#pragma once

#include "conquer/passes/quantisation/integer_patterns/utilities.h"
#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

#include <cmath>
#include <functional>
#include <algorithm>

namespace conquer::integer_quant {

template <typename OpTy>
struct QuantiseTablePattern : public mlir::RewritePattern {
    using MathFuncType = std::function<double(double)>;
    MathFuncType mathFunc;

    QuantiseTablePattern(mlir::MLIRContext *context, MathFuncType func)
        : RewritePattern(OpTy::getOperationName(), 1, context),
          mathFunc(std::move(func)) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
        L_TRACE("Applying QuantiseTablePattern to: " << compact(op));
        // 1. Look ahead: Find if this operation is consumed by a conquer.quantise
        mlir::Operation* quantOp = nullptr;
        for (auto *user : op->getUsers()) {
            if (user->getName().getStringRef() == "conquer.quantise") {
                quantOp = user;
                break;
            }
        }

        // If it's not being quantised, leave it as float.
        if (!quantOp) return mlir::failure();

        // 2. Check the target quantization type
        const auto outType = llvm::cast<mlir::RankedTensorType>(quantOp->getResult(0).getType()).getElementType();
        if (!outType.isIntOrIndex()) return mlir::failure();

        const unsigned bitWidth = outType.getIntOrFloatBitWidth();

        // 3. We only pull up into tables for i8 and i16. Leave i4 or i32 alone.
        if (bitWidth != 8 && bitWidth != 16) return mlir::failure();

        const mlir::Location loc = op->getLoc();

        // 4. Quantise Input Activation
        // We quantise the float input of the transcendental op, assuming MinMax calibration.
        const mlir::Value rawInput = op->getOperand(0);
        constexpr bool isSymmetric = false; // Transcendentals are usually highly asymmetric

        auto [quantA, aZpVal] = quantiseActivation(
            rewriter, op, rawInput, outType, outType, isSymmetric, CalibrationMethod::MinMax
        );
        if (!quantA) return mlir::failure();

        const double sA = getScalesFromValue(quantA)[0];

        // 5. Extract Output Scale & Zero Point directly from the conquer.quantise node!
        const auto scalesAttr = quantOp->getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
        const auto zpAttr = quantOp->getAttrOfType<mlir::IntegerAttr>("zero_point");

        double outScale = scalesAttr ? scalesAttr.asArrayRef()[0] : 1.0;
        int64_t outZp = zpAttr ? zpAttr.getInt() : 0;
        if (outScale <= 0.0) outScale = 1.0;

        // Safe rounding helper to prevent C++ UB on float-to-int cast with NaNs/Infs
        auto safe_round = [](const double val) -> int64_t {
            if (std::isnan(val)) return 0;
            if (std::isinf(val)) return val > 0 ? 32767 : -32768;
            return static_cast<int64_t>(std::round(val));
        };

        mlir::Value finalIntOut;

        // 6. Generate the i8 Table (256 entries)
        if (bitWidth == 8) {
            std::vector<int8_t> tableData(256);
            for (int i = -128; i <= 127; ++i) {
                const double x_float = (i - aZpVal) * sA;
                const double y_float = mathFunc(x_float);
                int64_t q_y = safe_round(y_float / outScale) + outZp;

                tableData[i + 128] = static_cast<int8_t>(std::clamp<int64_t>(q_y, -128, 127));
            }

            const auto tableType = mlir::RankedTensorType::get({256}, rewriter.getI8Type());
            const auto tableAttr = mlir::DenseElementsAttr::get(tableType, llvm::ArrayRef<int8_t>(tableData));
            const auto tableConst = mlir::tosa::ConstOp::create(rewriter, loc, tableType, tableAttr).getResult();

            const auto outShape = llvm::cast<mlir::RankedTensorType>(quantOp->getResult(0).getType()).getShape();
            const auto resType = mlir::RankedTensorType::get(outShape, rewriter.getI8Type());

            auto tableOp = mlir::tosa::TableOp::create(rewriter, loc, resType, quantA, tableConst);
            tableOp->setAttr("conquer.int.quantised", rewriter.getUnitAttr());

            finalIntOut = tableOp.getResult();
        }
        // 7. Generate the i16 Table (513 entries, 9.7 fixed point domain)
        else if (bitWidth == 16) {
            std::vector<int16_t> tableData(513);
            for (int i = 0; i < 513; ++i) {
                // TOSA 16-bit tables index using the top 9 bits: index = (val + 32768) >> 7.
                // We reverse this to find the integer value that corresponds to this exact index.
                const int32_t x_int = (i << 7) - 32768;

                const double x_float = (x_int - aZpVal) * sA;
                const double y_float = mathFunc(x_float);

                // We bake the output ZP directly into the table!
                int64_t q_y = safe_round(y_float / outScale) + outZp;
                tableData[i] = static_cast<int16_t>(std::clamp<int64_t>(q_y, -32768, 32767));
            }

            const auto tableType = mlir::RankedTensorType::get({513}, rewriter.getI16Type());
            const auto tableAttr = mlir::DenseElementsAttr::get(tableType, llvm::ArrayRef<int16_t>(tableData));
            const auto tableConst = mlir::tosa::ConstOp::create(rewriter, loc, tableType, tableAttr).getResult();

            const auto outShape = llvm::cast<mlir::RankedTensorType>(quantOp->getResult(0).getType()).getShape();

            // i16 table natively outputs an interpolated i32 value scaled by << 7
            const auto tableOutType = mlir::RankedTensorType::get(outShape, rewriter.getI32Type());
            auto tableOp = mlir::tosa::TableOp::create(rewriter, loc, tableOutType, quantA, tableConst);
            tableOp->setAttr("conquer.int.quantised", rewriter.getUnitAttr());

            // Because the output is scaled by << 7, we must shift it back down (>> 7).
            // A multiplier of (1<<30) and a shift of 37 perfectly equals division by 128!
            // ZPs are 0 here because the ZP is ALREADY baked into the interpolated table values.
            finalIntOut = createTosaRescale(rewriter, loc, tableOp.getResult(),
                                            (1 << 30), 37, 0, 0, outType);
        }

        // 8. Replace the conquer.quantise op with the Table Output
        rewriter.replaceOp(quantOp, finalIntOut);

        // 9. Clean up the original transcendental op if it has no more uses
        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }

        return mlir::success();
    }
};

// ==============================================================================
// POPULATE METHOD
// ==============================================================================
inline void populateTablePatterns(mlir::RewritePatternSet &patterns) {
    mlir::MLIRContext *context = patterns.getContext();

    // Standard Transcendentals
    patterns.add<QuantiseTablePattern<mlir::tosa::ExpOp>>(context, [](const double x) { return std::exp(x); });
    patterns.add<QuantiseTablePattern<mlir::tosa::CosOp>>(context, [](const double x) { return std::cos(x); });
    patterns.add<QuantiseTablePattern<mlir::tosa::SinOp>>(context, [](const double x) { return std::sin(x); });

    // Logarithms (Safeguarded against <= 0)
    patterns.add<QuantiseTablePattern<mlir::tosa::LogOp>>(context, [](const double x) {
        return std::log(std::max(x, 1e-7));
    });

    // Inverses (Safeguarded against div by zero)
    patterns.add<QuantiseTablePattern<mlir::tosa::ReciprocalOp>>(context, [](const double x) {
        return 1.0 / (std::abs(x) < 1e-7 ? (x < 0 ? -1e-7 : 1e-7) : x);
    });
    patterns.add<QuantiseTablePattern<mlir::tosa::RsqrtOp>>(context, [](const double x) {
        return 1.0 / std::sqrt(std::max(x, 1e-7));
    });

    // Activations
    patterns.add<QuantiseTablePattern<mlir::tosa::SigmoidOp>>(context, [](const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
    patterns.add<QuantiseTablePattern<mlir::tosa::TanhOp>>(context, [](const double x) {
        return std::tanh(x);
    });
    patterns.add<QuantiseTablePattern<mlir::tosa::ErfOp>>(context, [](const double x) {
        return std::erf(x);
    });
}

} // namespace conquer::integer_quant