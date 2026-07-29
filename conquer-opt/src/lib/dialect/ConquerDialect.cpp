#include "conquer/dialect/ConquerDialect.h"

#include "ConquerDialect.cpp.inc"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/SmallVector.h>

#include <cmath>
#include <optional>

void conquer::ConquerDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ConquerOps.cpp.inc"
        >();
}

#define GET_OP_CLASSES
#include "ConquerOps.cpp.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

enum class PartitionKind {
    PerTensor,
    Axis,
    DepthwiseCM
};

static std::optional<PartitionKind> parsePartitionKind(mlir::StringAttr attr) {
    if (!attr)
        return std::nullopt;

    const mlir::StringRef s = attr.getValue();
    if (s == "per_tensor")
        return PartitionKind::PerTensor;
    if (s == "axis")
        return PartitionKind::Axis;
    if (s == "depthwise_cm")
        return PartitionKind::DepthwiseCM;

    return std::nullopt;
}

static int64_t storageMinForWidth(const unsigned width) {
    return -(int64_t{1} << (width - 1));
}

static int64_t storageMaxForWidth(const unsigned width) {
    return (int64_t{1} << (width - 1)) - 1;
}

static int64_t clampMinForWidth(const unsigned width, const bool narrowRange) {
    const int64_t qmin = storageMinForWidth(width);
    return narrowRange ? qmin + 1 : qmin;
}

static bool sameQParams(mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->getAttr("scales") == rhs->getAttr("scales") &&
           lhs->getAttr("zero_point") == rhs->getAttr("zero_point") &&
           lhs->getAttr("partition_kind") == rhs->getAttr("partition_kind") &&
           lhs->getAttr("axis") == rhs->getAttr("axis") &&
           lhs->getAttr("narrow_range") == rhs->getAttr("narrow_range");
}

static mlir::LogicalResult verifyScaleCountForShape(mlir::Operation *op,
                                                    llvm::ArrayRef<int64_t> shape,
                                                    PartitionKind kind,
                                                    std::optional<int64_t> axis,
                                                    size_t scaleCount) {
    switch (kind) {
    case PartitionKind::PerTensor:
        if (scaleCount != 1)
            return op->emitOpError() << "requires exactly 1 scale for partition_kind = \"per_tensor\"";
        return mlir::success();

    case PartitionKind::Axis:
        if (!axis.has_value())
            return op->emitOpError() << "requires 'axis' for partition_kind = \"axis\"";

        if (*axis < 0 || *axis >= static_cast<int64_t>(shape.size()))
            return op->emitOpError() << "'axis' out of range for tensor rank " << shape.size();

        if (!mlir::ShapedType::isDynamic(shape[*axis]) &&
            scaleCount != static_cast<size_t>(shape[*axis])) {
            return op->emitOpError()
                   << "scale count (" << scaleCount << ") does not match axis dimension ("
                   << shape[*axis] << ")";
        }
        return mlir::success();

    case PartitionKind::DepthwiseCM:
        if (shape.size() != 4)
            return op->emitOpError()
                   << "requires rank-4 tensor for partition_kind = \"depthwise_cm\"";

        if (!mlir::ShapedType::isDynamic(shape[2]) &&
            !mlir::ShapedType::isDynamic(shape[3])) {
            const int64_t expected = shape[2] * shape[3];
            if (scaleCount != static_cast<size_t>(expected)) {
                return op->emitOpError()
                       << "scale count (" << scaleCount
                       << ") does not match depthwise C*M channel count (" << expected << ")";
            }
        }
        return mlir::success();
    }

    return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult verifyQuantLikeOp(OpTy op, mlir::RankedTensorType storageTensorType) {
    auto *operation = op.getOperation();

    const auto scalesAttr = operation->template getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
    const auto zpAttr = operation->template getAttrOfType<mlir::IntegerAttr>("zero_point");
    const auto partitionKindAttr = operation->template getAttrOfType<mlir::StringAttr>("partition_kind");
    const auto axisAttr = operation->template getAttrOfType<mlir::IntegerAttr>("axis");
    const auto narrowRangeAttr = operation->template getAttrOfType<mlir::BoolAttr>("narrow_range");

    if (!scalesAttr)
        return op.emitOpError() << "requires 'scales' attribute";

    const auto scales = scalesAttr.asArrayRef();
    if (scales.empty())
        return op.emitOpError() << "'scales' must be non-empty";

    for (double s : scales) {
        if (!std::isfinite(s) || s <= 0.0)
            return op.emitOpError() << "all scales must be finite and > 0";
    }

    if (!zpAttr)
        return op.emitOpError() << "requires 'zero_point' attribute";

    if (!partitionKindAttr)
        return op.emitOpError() << "requires 'partition_kind' attribute";

    const auto kind = parsePartitionKind(partitionKindAttr);
    if (!kind.has_value())
        return op.emitOpError()
               << "invalid partition_kind, expected one of: "
                  "\"per_tensor\", \"axis\", \"depthwise_cm\"";

    const auto elemTy = llvm::cast<mlir::IntegerType>(storageTensorType.getElementType());
    const unsigned width = elemTy.getWidth();

    const int64_t zp = zpAttr.getInt();
    const int64_t storageMin = storageMinForWidth(width);
    const int64_t storageMax = storageMaxForWidth(width);
    if (zp < storageMin || zp > storageMax) {
        return op.emitOpError()
               << "'zero_point' (" << zp << ") out of signed storage range ["
               << storageMin << ", " << storageMax << "] for i" << width;
    }

    const std::optional<int64_t> axis =
        axisAttr ? std::optional<int64_t>(axisAttr.getInt()) : std::nullopt;

    if (*kind == PartitionKind::PerTensor && axis.has_value())
        return op.emitOpError() << "'axis' must be absent for partition_kind = \"per_tensor\"";

    if (*kind == PartitionKind::DepthwiseCM && axis.has_value())
        return op.emitOpError() << "'axis' must be absent for partition_kind = \"depthwise_cm\"";

    (void)narrowRangeAttr; // Present due default attr in ODS; no extra structural check needed here.

    return verifyScaleCountForShape(operation, storageTensorType.getShape(), *kind, axis, scales.size());
}

static int64_t getChannelForFlatIndex(llvm::ArrayRef<int64_t> shape,
                                      const int64_t flatIndex,
                                      PartitionKind kind,
                                      std::optional<int64_t> axis) {
    switch (kind) {
    case PartitionKind::PerTensor:
        return 0;

    case PartitionKind::Axis: {
        const int64_t ax = *axis;
        int64_t stride = 1;
        for (size_t d = static_cast<size_t>(ax) + 1; d < shape.size(); ++d)
            stride *= shape[d];
        return (flatIndex / stride) % shape[ax];
    }

    case PartitionKind::DepthwiseCM: {
        // shape = [KH, KW, C, M], channels = C * M
        const int64_t C = shape[2];
        const int64_t M = shape[3];

        const int64_t m = flatIndex % M;
        const int64_t tmp = flatIndex / M;
        const int64_t c = tmp % C;
        return c * M + m;
    }
    }

    return 0;
}

static std::optional<std::pair<mlir::Operation *, mlir::DenseElementsAttr>>
getDenseConstDefiningOp(mlir::Value value) {
    if (auto tosaConst = value.getDefiningOp<mlir::tosa::ConstOp>()) {
        if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(tosaConst.getValues()))
            return std::make_pair(tosaConst.getOperation(), dense);
    }

    if (auto arithConst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(arithConst.getValue()))
            return std::make_pair(arithConst.getOperation(), dense);
    }

    return std::nullopt;
}

static mlir::Value buildConstLike(mlir::Operation *sourceConst,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Location loc,
                                  mlir::DenseElementsAttr valueAttr) {
    if (llvm::isa<mlir::tosa::ConstOp>(sourceConst))
        return mlir::tosa::ConstOp::create(rewriter, loc, static_cast<mlir::Type>(valueAttr.getType()), valueAttr).getResult();

    return mlir::arith::ConstantOp::create(rewriter, loc, valueAttr).getResult();
}

static mlir::FailureOr<mlir::DenseElementsAttr>
foldQuantiseDenseConst(conquer::QuantiseOp op, mlir::DenseElementsAttr inputAttr) {
    const auto outType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    const auto outElemTy = llvm::cast<mlir::IntegerType>(outType.getElementType());
    const unsigned width = outElemTy.getWidth();

    const auto scalesAttr = op->getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
    const auto scales = scalesAttr.asArrayRef();
    const int64_t zp = op->getAttrOfType<mlir::IntegerAttr>("zero_point").getInt();
    const auto kind = *parsePartitionKind(op->getAttrOfType<mlir::StringAttr>("partition_kind"));
    const auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
    const std::optional<int64_t> axis =
        axisAttr ? std::optional<int64_t>(axisAttr.getInt()) : std::nullopt;
    const bool narrowRange =
        op->getAttrOfType<mlir::BoolAttr>("narrow_range").getValue();

    if (mlir::failed(verifyScaleCountForShape(op.getOperation(), outType.getShape(), kind, axis, scales.size())))
        return mlir::failure();

    const int64_t qmin = clampMinForWidth(width, narrowRange);
    const int64_t qmax = storageMaxForWidth(width);

    llvm::SmallVector<llvm::APInt> quantisedValues;
    quantisedValues.reserve(inputAttr.getNumElements());

    int64_t flatIndex = 0;
    for (const llvm::APFloat &apf : inputAttr.getValues<llvm::APFloat>()) {
        const double real = apf.convertToDouble();
        const int64_t channel =
            getChannelForFlatIndex(outType.getShape(), flatIndex, kind, axis);

        const double scaled = real / scales[channel];
        int64_t q = static_cast<int64_t>(std::nearbyint(scaled)) + zp;
        q = std::clamp(q, qmin, qmax);

        quantisedValues.emplace_back(width, static_cast<uint64_t>(q), /*isSigned=*/true);
        ++flatIndex;
    }

    return mlir::DenseElementsAttr::get(outType, quantisedValues);
}

static mlir::FailureOr<mlir::DenseElementsAttr>
foldDequantiseDenseConst(conquer::DequantiseOp op, mlir::DenseElementsAttr inputAttr) {
    const auto outType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    const auto outElemTy = llvm::cast<mlir::FloatType>(outType.getElementType());

    const auto scalesAttr = op->getAttrOfType<mlir::DenseF64ArrayAttr>("scales");
    const auto scales = scalesAttr.asArrayRef();
    const int64_t zp = op->getAttrOfType<mlir::IntegerAttr>("zero_point").getInt();
    const auto kind =
        *parsePartitionKind(op->getAttrOfType<mlir::StringAttr>("partition_kind"));
    const auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
    const std::optional<int64_t> axis =
        axisAttr ? std::optional<int64_t>(axisAttr.getInt()) : std::nullopt;

    const auto inType = llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    if (mlir::failed(verifyScaleCountForShape(op.getOperation(), inType.getShape(), kind, axis, scales.size())))
        return mlir::failure();

    llvm::SmallVector<llvm::APFloat> dequantisedValues;
    dequantisedValues.reserve(inputAttr.getNumElements());

    const llvm::fltSemantics &targetSemantics = outElemTy.getFloatSemantics();

    int64_t flatIndex = 0;
    for (const llvm::APInt &api : inputAttr.getValues<llvm::APInt>()) {
        const int64_t q = api.getSExtValue();
        const int64_t channel =
            getChannelForFlatIndex(inType.getShape(), flatIndex, kind, axis);

        const double real = scales[channel] * static_cast<double>(q - zp);

        llvm::APFloat apf(real);
        bool losesInfo = false;
        apf.convert(targetSemantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        (void)losesInfo;

        dequantisedValues.push_back(apf);
        ++flatIndex;
    }

    return mlir::DenseElementsAttr::get(outType, dequantisedValues);
}

//===----------------------------------------------------------------------===//
// Canonicalisation patterns
//===----------------------------------------------------------------------===//

struct FoldQuantiseOfMatchingDequantisePattern
    : mlir::OpRewritePattern<conquer::QuantiseOp> {
    using OpRewritePattern<conquer::QuantiseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(conquer::QuantiseOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        auto dq = op.getInput().getDefiningOp<conquer::DequantiseOp>();
        if (!dq)
            return mlir::failure();

        if (!sameQParams(op.getOperation(), dq.getOperation()))
            return mlir::failure();

        if (op.getResult().getType() != dq.getInput().getType())
            return mlir::failure();

        rewriter.replaceOp(op, dq.getInput());
        return mlir::success();
    }
};

struct FoldQuantiseConstPattern : mlir::OpRewritePattern<conquer::QuantiseOp> {
    using OpRewritePattern<conquer::QuantiseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(conquer::QuantiseOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        auto maybeConst = getDenseConstDefiningOp(op.getInput());
        if (!maybeConst)
            return mlir::failure();

        auto [sourceConst, denseAttr] = *maybeConst;

        auto folded = foldQuantiseDenseConst(op, denseAttr);
        if (mlir::failed(folded))
            return mlir::failure();

        rewriter.replaceOp(op, buildConstLike(sourceConst, rewriter, op.getLoc(), *folded));
        return mlir::success();
    }
};

struct FoldDequantiseConstPattern : mlir::OpRewritePattern<conquer::DequantiseOp> {
    using OpRewritePattern<conquer::DequantiseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(conquer::DequantiseOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        auto maybeConst = getDenseConstDefiningOp(op.getInput());
        if (!maybeConst)
            return mlir::failure();

        auto [sourceConst, denseAttr] = *maybeConst;

        auto folded = foldDequantiseDenseConst(op, denseAttr);
        if (mlir::failed(folded))
            return mlir::failure();

        rewriter.replaceOp(op, buildConstLike(sourceConst, rewriter, op.getLoc(), *folded));
        return mlir::success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuantiseOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult conquer::QuantiseOp::verify() {
    const auto storageTensorType =
        llvm::cast<mlir::RankedTensorType>(getResult().getType());
    return verifyQuantLikeOp(*this, storageTensorType);
}

void conquer::QuantiseOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                      mlir::MLIRContext *context) {
    results.add<
        FoldQuantiseOfMatchingDequantisePattern,
        FoldQuantiseConstPattern
    >(context);
}

//===----------------------------------------------------------------------===//
// DequantiseOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult conquer::DequantiseOp::verify() {
    const auto storageTensorType =
        llvm::cast<mlir::RankedTensorType>(getInput().getType());
    return verifyQuantLikeOp(*this, storageTensorType);
}

void conquer::DequantiseOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                        mlir::MLIRContext *context) {
    // results.add<
    //     FoldDequantiseConstPattern
    // >(context);
}
