#include "conquer/passes/quantisation/lower_quantisation_pass.h"
#include "conquer/dialect/ConquerDialect.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>
#include <vector>

namespace conquer {
namespace {

// ==============================================================================
// Helpers
// ==============================================================================

// Returns a broadcast shape compatible with the partitioning scheme used by
// conquer.quantise / conquer.dequantise.
//
// Supported partition kinds:
//   - per_tensor   => [1, 1, ..., 1]
//   - axis         => [1, ..., dim(axis), ..., 1]
//   - depthwise_cm => [1, 1, C, M] for rank-4 [KH, KW, C, M]
[[nodiscard]] std::optional<llvm::SmallVector<int64_t>>
getBroadcastShape(const mlir::RankedTensorType tensorType,
                  const mlir::StringRef partitionKind,
                  const std::optional<int64_t> axis) {
    llvm::SmallVector<int64_t> shape(tensorType.getRank(), 1);

    if (partitionKind == "per_tensor") {
        return shape;
    }

    if (partitionKind == "axis") {
        if (!axis.has_value()) return std::nullopt;

        const int64_t axisIdx = *axis;
        if (axisIdx < 0 || axisIdx >= tensorType.getRank()) return std::nullopt;

        const int64_t axisDim = tensorType.getDimSize(axisIdx);
        if (mlir::ShapedType::isDynamic(axisDim)) return std::nullopt;

        shape[static_cast<size_t>(axisIdx)] = axisDim;
        return shape;
    }

    if (partitionKind == "depthwise_cm") {
        if (tensorType.getRank() != 4) return std::nullopt;

        const int64_t cDim = tensorType.getDimSize(2);
        const int64_t mDim = tensorType.getDimSize(3);
        if (mlir::ShapedType::isDynamic(cDim) || mlir::ShapedType::isDynamic(mDim)) {
            return std::nullopt;
        }

        shape[2] = cDim;
        shape[3] = mDim;
        return shape;
    }

    return std::nullopt;
}

// Calculates the strictly bounded signed storage min/max.
// Wide integer paths such as i48 are supported.
void getQMinMax(const unsigned bitWidth, const bool narrowRange, int64_t &qmin, int64_t &qmax) {
    qmin = -(int64_t{1} << (bitWidth - 1));
    qmax =  (int64_t{1} << (bitWidth - 1)) - 1;
    if (narrowRange) {
        qmin += 1;
    }
}

// Select a legal float arithmetic type for lowering quantise/dequantise math.
// - fp8 arithmetic is widened to f16
// - wide integer quant/dequant arithmetic is widened to f32
[[nodiscard]] mlir::Type getLegalArithmeticFloatType(mlir::MLIRContext *ctx,
                                                     const mlir::Type boundaryFloatType,
                                                     const unsigned integerBitWidth) {
    mlir::Type arithType = boundaryFloatType;

    if (!mlir::isa<mlir::FloatType>(arithType)) {
        arithType = mlir::Float32Type::get(ctx);
    } else if (arithType.isF8E5M2() || arithType.isF8E4M3FN()) {
        arithType = mlir::Float16Type::get(ctx);
    }

    if (integerBitWidth > 16 && !arithType.isF32()) {
        arithType = mlir::Float32Type::get(ctx);
    }

    return arithType;
}

[[nodiscard]] mlir::RankedTensorType withElementType(const mlir::RankedTensorType type, const mlir::Type elemType) {
    return mlir::RankedTensorType::get(type.getShape(), elemType);
}

[[nodiscard]] mlir::Value castTensorToElementType(mlir::PatternRewriter &rewriter, const mlir::Location loc,
                                                  const mlir::Value value, const mlir::Type elemType) {
    const auto valueType = llvm::cast<mlir::RankedTensorType>(value.getType());
    const mlir::Type currentElemType = valueType.getElementType();
    if (currentElemType == elemType) return value;

    const bool current_is_16bit = currentElemType.isF16() || currentElemType.isBF16();
    const bool target_is_16bit = elemType.isF16() || elemType.isBF16();

    if (current_is_16bit && target_is_16bit) {
        const auto f32Type = withElementType(valueType, rewriter.getF32Type());
        auto bridge_cast = mlir::tosa::CastOp::create(rewriter, loc, f32Type, value);
        const auto targetType = withElementType(valueType, elemType);
        return mlir::tosa::CastOp::create(rewriter, loc, targetType, bridge_cast.getResult()).getResult();
    }

    const auto targetType = withElementType(valueType, elemType);
    return mlir::tosa::CastOp::create(rewriter, loc, targetType, value).getResult();
}

// Safely builds a DenseElementsAttr for float tensors without triggering MLIR
// assertions for lower-precision element types.
[[nodiscard]] mlir::DenseElementsAttr getFloatArrayAttr(const mlir::RankedTensorType type,
                                                        const std::vector<float> &values) {
    const mlir::Type elemType = type.getElementType();

    if (elemType.isF32()) {
        return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(values));
    }

    const auto floatType = llvm::cast<mlir::FloatType>(elemType);
    const llvm::fltSemantics &sem = floatType.getFloatSemantics();

    llvm::SmallVector<llvm::APFloat> apValues;
    apValues.reserve(values.size());
    for (const float v : values) {
        bool losesInfo = false;
        llvm::APFloat apv(v);
        apv.convert(sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        apValues.push_back(apv);
    }

    return mlir::DenseElementsAttr::get(type, apValues);
}

// Creates a 0-valued i8 tensor for the modern tosa.mul shift parameter.
[[nodiscard]] mlir::Value createZeroShiftConst(mlir::PatternRewriter &rewriter, const mlir::Location loc) {
    const auto shiftType = mlir::RankedTensorType::get({1}, rewriter.getI8Type());
    const auto shiftAttr = mlir::DenseElementsAttr::get(shiftType, llvm::ArrayRef<int8_t>{0});
    return mlir::tosa::ConstOp::create(rewriter, loc, shiftType, shiftAttr).getResult();
}

// ==============================================================================
// 1. Lower conquer.quantise (Float -> Int)
// TOSA Math:
//   q = cast_target(clamp((cast_or_widen(x) * (1/scale)) + zp, qmin, qmax))
// ==============================================================================
struct LowerQuantiseOpPattern : public mlir::OpRewritePattern<conquer::QuantiseOp> {
    using OpRewritePattern<conquer::QuantiseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(conquer::QuantiseOp op, mlir::PatternRewriter &rewriter) const override {
        const mlir::Location loc = op.getLoc();
        const mlir::Value input = op.getInput();

        const auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        const auto outType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
        const auto outElemType = llvm::cast<mlir::IntegerType>(outType.getElementType());

        const auto scalesAttr = op->getAttrOfType<mlir::DenseF64ArrayAttr>("scales").asArrayRef();
        const int64_t zp = op->getAttrOfType<mlir::IntegerAttr>("zero_point").getInt();
        const auto pk = op->getAttrOfType<mlir::StringAttr>("partition_kind").getValue();
        const auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
        const std::optional<int64_t> axis = axisAttr ? std::optional<int64_t>(axisAttr.getInt()) : std::nullopt;
        const bool narrowRange = op->getAttrOfType<mlir::BoolAttr>("narrow_range").getValue();

        const auto broadcastShapeOpt = getBroadcastShape(inputType, pk, axis);
        if (!broadcastShapeOpt.has_value()) return mlir::failure();
        const auto &broadcastShape = *broadcastShapeOpt;

        const mlir::Type arithElemType =
            getLegalArithmeticFloatType(rewriter.getContext(), inputType.getElementType(), outElemType.getWidth());

        const auto arithInputType = withElementType(inputType, arithElemType);
        const auto arithOutType = withElementType(outType, arithElemType);

        const mlir::Value arithInput = castTensorToElementType(rewriter, loc, input, arithElemType);

        // 1. Prepare inverse scales in arithmetic float type.
        std::vector<float> invScales(scalesAttr.size());
        for (size_t i = 0; i < scalesAttr.size(); ++i) {
            invScales[i] = 1.0f / static_cast<float>(scalesAttr[i]);
        }

        const auto scaleType = mlir::RankedTensorType::get(broadcastShape, arithElemType);
        auto invScaleConst = mlir::tosa::ConstOp::create(
            rewriter, loc, scaleType, getFloatArrayAttr(scaleType, invScales));

        // 2. Prepare zero-point constant in arithmetic float type.
        const auto zpType = mlir::RankedTensorType::get(broadcastShape, arithElemType);
        std::vector<float> zps(scalesAttr.size(), static_cast<float>(zp));
        auto zpConst = mlir::tosa::ConstOp::create(
            rewriter, loc, zpType, getFloatArrayAttr(zpType, zps));

        // 3. Math: Mul -> Add -> Clamp -> Cast(target int type)
        const mlir::Value shiftConst = createZeroShiftConst(rewriter, loc);

        auto mulOp = mlir::tosa::MulOp::create(
            rewriter, loc, arithInputType, arithInput, invScaleConst.getResult(), shiftConst);

        auto addOp = mlir::tosa::AddOp::create(
            rewriter, loc, arithInputType, mulOp.getResult(), zpConst.getResult());

        int64_t qmin = 0, qmax = 0;
        getQMinMax(outElemType.getWidth(), narrowRange, qmin, qmax);

        const auto minAttr = rewriter.getFloatAttr(arithElemType, static_cast<double>(qmin));
        const auto maxAttr = rewriter.getFloatAttr(arithElemType, static_cast<double>(qmax));

        auto clampOp = mlir::tosa::ClampOp::create(
            rewriter, loc, arithInputType, addOp.getResult(), minAttr, maxAttr);

        auto finalCastOp = mlir::tosa::CastOp::create(
            rewriter, loc, outType, clampOp.getResult());

        rewriter.replaceOp(op, finalCastOp.getResult());
        (void)arithOutType; // kept for symmetry/documentation
        return mlir::success();
    }
};

// ==============================================================================
// 2. Lower conquer.dequantise (Int -> Float)
// TOSA Math:
//   x = (cast_or_widen_float(q) - zp_float) * scale
//   if needed, cast back down to the requested float element type
//
// This avoids truncating wide integer inputs such as i48 through i32.
// ==============================================================================
struct LowerDequantiseOpPattern : public mlir::OpRewritePattern<conquer::DequantiseOp> {
    using OpRewritePattern<conquer::DequantiseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(conquer::DequantiseOp op, mlir::PatternRewriter &rewriter) const override {
        const mlir::Location loc = op.getLoc();
        const mlir::Value input = op.getInput();

        const auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        const auto outType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
        const auto inputElemType = llvm::cast<mlir::IntegerType>(inputType.getElementType());

        const auto scalesAttr = op->getAttrOfType<mlir::DenseF64ArrayAttr>("scales").asArrayRef();
        const int64_t zp = op->getAttrOfType<mlir::IntegerAttr>("zero_point").getInt();
        const auto pk = op->getAttrOfType<mlir::StringAttr>("partition_kind").getValue();
        const auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
        const std::optional<int64_t> axis = axisAttr ? std::optional<int64_t>(axisAttr.getInt()) : std::nullopt;

        const auto broadcastShapeOpt = getBroadcastShape(inputType, pk, axis);
        if (!broadcastShapeOpt.has_value()) return mlir::failure();
        const auto &broadcastShape = *broadcastShapeOpt;

        const mlir::Type arithElemType =
            getLegalArithmeticFloatType(rewriter.getContext(), outType.getElementType(), inputElemType.getWidth());
        const auto arithOutType = withElementType(outType, arithElemType);

        // 1. Prepare scale constant in arithmetic float type.
        std::vector<float> scales(scalesAttr.size());
        for (size_t i = 0; i < scalesAttr.size(); ++i) {
            scales[i] = static_cast<float>(scalesAttr[i]);
        }

        const auto scaleType = mlir::RankedTensorType::get(broadcastShape, arithElemType);
        auto scaleConst = mlir::tosa::ConstOp::create(
            rewriter, loc, scaleType, getFloatArrayAttr(scaleType, scales));

        // 2. Prepare zero-point constant in arithmetic float type.
        const auto zpType = mlir::RankedTensorType::get(broadcastShape, arithElemType);
        std::vector<float> zps(scalesAttr.size(), static_cast<float>(zp));
        auto zpConst = mlir::tosa::ConstOp::create(
            rewriter, loc, zpType, getFloatArrayAttr(zpType, zps));

        // 3. Math: Cast(Float) -> Sub(Float) -> Mul(Float)
        auto castFloatOp = mlir::tosa::CastOp::create(
            rewriter, loc, arithOutType, input);

        auto subOp = mlir::tosa::SubOp::create(
            rewriter, loc, arithOutType, castFloatOp.getResult(), zpConst.getResult());

        const mlir::Value shiftConst = createZeroShiftConst(rewriter, loc);
        auto mulOp = mlir::tosa::MulOp::create(
            rewriter, loc, arithOutType, subOp.getResult(), scaleConst.getResult(), shiftConst);

        mlir::Value finalOut = mulOp.getResult();
        if (arithElemType != outType.getElementType()) {
            finalOut = castTensorToElementType(rewriter, loc, finalOut, outType.getElementType());
        }

        rewriter.replaceOp(op, finalOut);
        return mlir::success();
    }
};

} // namespace

void LowerQuantisationPass::getDependentDialects(mlir::DialectRegistry &registry) const {
    registry.insert<mlir::tosa::TosaDialect>();
    registry.insert<conquer::ConquerDialect>();
}

void LowerQuantisationPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerQuantiseOpPattern, LowerDequantiseOpPattern>(&getContext());

    if (mlir::failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace conquer
