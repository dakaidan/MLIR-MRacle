#include "conquer/passes/analysis/weight_analysis_pass.h"
#include "conquer/core/logging.h"

#include "conquer/quantisation/stats.h"
#include "conquer/quantisation/tosa_utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Parallel.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>
#include <mlir/IR/DialectResourceBlobManager.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-weight-analysis"

namespace {

struct ResolvedConstTensor {
    mlir::Operation *rootConstOp = nullptr;
    mlir::ElementsAttr rootTensorAttr;     // raw root constant
    mlir::ElementsAttr resolvedTensorAttr; // tensor as seen by the consumer
};

static mlir::ElementsAttr getConstTensorAttr(mlir::Operation *defOp) {
    if (auto tosaConst = llvm::dyn_cast<mlir::tosa::ConstOp>(defOp)) {
        return llvm::dyn_cast<mlir::ElementsAttr>(tosaConst.getValues());
    }
    if (auto arithConst = llvm::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
        return llvm::dyn_cast<mlir::ElementsAttr>(arithConst.getValue());
    }
    return {};
}

static llvm::StringRef partitionKindToString(const conquer::WeightPartitionKind kind) {
    switch (kind) {
    case conquer::WeightPartitionKind::Axis:
        return "axis";
    case conquer::WeightPartitionKind::DepthwiseCM:
        return "depthwise_cm";
    case conquer::WeightPartitionKind::None:
        return "none";
    }
    return "none";
}

static bool extractFloatValues(const mlir::ElementsAttr attr, std::vector<float> &values) {
    if (!mlir::isa<mlir::FloatType>(attr.getElementType()))
        return false;

    values.clear();
    values.reserve(attr.getNumElements());

    if (const auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        for (const llvm::APFloat &apf : denseAttr.getValues<llvm::APFloat>()) {
            values.push_back(apf.convertToFloat());
        }
        return true;
    }

    if (const auto resourceAttr = llvm::dyn_cast<mlir::DenseResourceElementsAttr>(attr)) {
        const auto blob = resourceAttr.getRawHandle().getBlob();
        if (!blob) return false;

        const auto elementType = attr.getElementType();
        if (elementType.isF32()) {
            const auto data = blob->getDataAs<float>();
            values.assign(data.begin(), data.end());
            return true;
        }
        if (elementType.isF16()) {
            const auto data = blob->getDataAs<uint16_t>();
            for (const uint16_t raw : data) {
                llvm::APFloat apf(llvm::APFloat::IEEEhalf(), llvm::APInt(16, raw));
                values.push_back(apf.convertToFloat());
            }
            return true;
        }
        if (elementType.isBF16()) {
            const auto data = blob->getDataAs<uint16_t>();
            for (const uint16_t raw : data) {
                llvm::APFloat apf(llvm::APFloat::BFloat(), llvm::APInt(16, raw));
                values.push_back(apf.convertToFloat());
            }
            return true;
        }
    }

    return false;
}

static std::optional<llvm::SmallVector<int64_t>> extractI64Values(const mlir::ElementsAttr attr) {
    if (!attr.getElementType().isIntOrIndex())
        return std::nullopt;

    llvm::SmallVector<int64_t> values;
    values.reserve(attr.getNumElements());

    if (const auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        for (const llvm::APInt &api : dense.getValues<llvm::APInt>()) {
            values.push_back(api.getSExtValue());
        }
        return values;
    }

    if (const auto resource = llvm::dyn_cast<mlir::DenseResourceElementsAttr>(attr)) {
        const auto blob = resource.getRawHandle().getBlob();
        if (!blob) return std::nullopt;

        const auto elemType = attr.getElementType();
        if (elemType.isInteger(32)) {
            const auto data = blob->getDataAs<int32_t>();
            for (const int32_t v : data) values.push_back(v);
            return values;
        }
        if (elemType.isInteger(64) || elemType.isIndex()) {
            const auto data = blob->getDataAs<int64_t>();
            for (const int64_t v : data) values.push_back(v);
            return values;
        }
    }
    return std::nullopt;
}

static std::optional<llvm::SmallVector<int64_t>>
getI64ArrayAttr(mlir::Operation *op, const llvm::ArrayRef<llvm::StringRef> candidateNames) {
    for (const llvm::StringRef name : candidateNames) {
        if (const auto dense = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name)) {
            llvm::SmallVector<int64_t> out(dense.asArrayRef().begin(), dense.asArrayRef().end());
            return out;
        }
        if (const auto arr = op->getAttrOfType<mlir::ArrayAttr>(name)) {
            llvm::SmallVector<int64_t> out;
            out.reserve(arr.size());
            for (const mlir::Attribute a : arr) {
                const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(a);
                if (!intAttr)
                    return std::nullopt;
                out.push_back(intAttr.getInt());
            }
            return out;
        }
    }
    return std::nullopt;
}

static std::optional<int64_t>
getI64ScalarAttr(mlir::Operation *op, const llvm::ArrayRef<llvm::StringRef> candidateNames) {
    for (const llvm::StringRef name : candidateNames) {
        if (const auto intAttr = op->getAttrOfType<mlir::IntegerAttr>(name)) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

static llvm::SmallVector<int64_t> computeStrides(const llvm::ArrayRef<int64_t> shape) {
    llvm::SmallVector<int64_t> strides(shape.size(), 1);
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = running;
        running *= shape[i];
    }
    return strides;
}

static llvm::SmallVector<int64_t> unflattenIndex(const int64_t flatIndex,
                                                 const llvm::ArrayRef<int64_t> shape) {
    llvm::SmallVector<int64_t> index(shape.size(), 0);
    const auto strides = computeStrides(shape);

    int64_t remaining = flatIndex;
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = remaining / strides[i];
        remaining %= strides[i];
    }
    return index;
}

static int64_t flattenIndex(const llvm::ArrayRef<int64_t> index,
                            const llvm::ArrayRef<int64_t> shape) {
    const auto strides = computeStrides(shape);
    int64_t flat = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        flat += index[i] * strides[i];
    }
    return flat;
}

static bool isValidPermutation(const llvm::ArrayRef<int64_t> perm, const int64_t rank) {
    if (static_cast<int64_t>(perm.size()) != rank)
        return false;

    std::vector<bool> seen(rank, false);
    for (const int64_t p : perm) {
        if (p < 0 || p >= rank || seen[p])
            return false;
        seen[p] = true;
    }
    return true;
}

static std::optional<mlir::DenseElementsAttr>
createDenseElementsAttr(const std::vector<float>& values, const mlir::RankedTensorType outType) {
    llvm::SmallVector<llvm::APFloat> outValues;
    outValues.reserve(values.size());
    const auto outElemTy = mlir::cast<mlir::FloatType>(outType.getElementType());
    const llvm::fltSemantics &targetSem = outElemTy.getFloatSemantics();

    for (const float f : values) {
        llvm::APFloat apf(f);
        bool losesInfo;
        apf.convert(targetSem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        outValues.push_back(apf);
    }
    return mlir::DenseElementsAttr::get(outType, outValues);
}

static std::optional<mlir::DenseElementsAttr>
reshapeFloatDenseAttr(const mlir::ElementsAttr inputAttr, const mlir::RankedTensorType outType) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;
    if (inValues.size() != static_cast<size_t>(outType.getNumElements())) return std::nullopt;
    return createDenseElementsAttr(inValues, outType);
}

static std::optional<mlir::DenseElementsAttr>
castFloatDenseAttr(const mlir::ElementsAttr inputAttr, const mlir::RankedTensorType outType) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;
    if (inValues.size() != static_cast<size_t>(outType.getNumElements())) return std::nullopt;
    return createDenseElementsAttr(inValues, outType);
}

static std::optional<mlir::DenseElementsAttr>
transposeFloatDenseAttr(const mlir::ElementsAttr inputAttr,
                        const mlir::RankedTensorType outType,
                        const llvm::ArrayRef<int64_t> perm) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;

    const auto inType = llvm::cast<mlir::RankedTensorType>(inputAttr.getType());
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    if (!isValidPermutation(perm, static_cast<int64_t>(inShape.size()))) return std::nullopt;
    if (inShape.size() != outShape.size()) return std::nullopt;

    std::vector<float> outValues;
    outValues.reserve(inValues.size());

    for (int64_t outFlat = 0, e = outType.getNumElements(); outFlat < e; ++outFlat) {
        const auto outIdx = unflattenIndex(outFlat, outShape);
        llvm::SmallVector<int64_t> inIdx(inShape.size(), 0);
        for (size_t d = 0; d < outIdx.size(); ++d) {
            inIdx[perm[d]] = outIdx[d];
        }
        const int64_t inFlat = flattenIndex(inIdx, inShape);
        outValues.push_back(inValues[inFlat]);
    }

    return createDenseElementsAttr(outValues, outType);
}

static std::optional<mlir::DenseElementsAttr>
reverseFloatDenseAttr(const mlir::ElementsAttr inputAttr,
                      const mlir::RankedTensorType outType,
                      const int64_t axis) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;

    const auto inType = llvm::cast<mlir::RankedTensorType>(inputAttr.getType());
    const auto shape = inType.getShape();

    if (shape != outType.getShape()) return std::nullopt;
    if (axis < 0 || axis >= static_cast<int64_t>(shape.size())) return std::nullopt;

    std::vector<float> outValues;
    outValues.reserve(inValues.size());

    for (int64_t outFlat = 0, e = outType.getNumElements(); outFlat < e; ++outFlat) {
        auto inIdx = unflattenIndex(outFlat, shape);
        inIdx[axis] = shape[axis] - 1 - inIdx[axis];
        const int64_t inFlat = flattenIndex(inIdx, shape);
        outValues.push_back(inValues[inFlat]);
    }

    return createDenseElementsAttr(outValues, outType);
}

static std::optional<mlir::DenseElementsAttr>
sliceFloatDenseAttr(const mlir::ElementsAttr inputAttr,
                    const mlir::RankedTensorType outType,
                    const llvm::ArrayRef<int64_t> start,
                    const llvm::ArrayRef<int64_t> size) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;

    const auto inType = llvm::cast<mlir::RankedTensorType>(inputAttr.getType());
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    if (start.size() != inShape.size() || size.size() != inShape.size()) return std::nullopt;
    if (size.size() != outShape.size()) return std::nullopt;

    for (size_t i = 0; i < inShape.size(); ++i) {
        if (start[i] < 0 || size[i] < 0) return std::nullopt;
        if (start[i] + size[i] > inShape[i]) return std::nullopt;
        if (size[i] != outShape[i]) return std::nullopt;
    }

    std::vector<float> outValues;
    outValues.reserve(outType.getNumElements());

    for (int64_t outFlat = 0, e = outType.getNumElements(); outFlat < e; ++outFlat) {
        auto inIdx = unflattenIndex(outFlat, outShape);
        for (size_t i = 0; i < inIdx.size(); ++i) {
            inIdx[i] += start[i];
        }
        const int64_t inFlat = flattenIndex(inIdx, inShape);
        outValues.push_back(inValues[inFlat]);
    }

    return createDenseElementsAttr(outValues, outType);
}

static std::optional<mlir::DenseElementsAttr>
tileFloatDenseAttr(const mlir::ElementsAttr inputAttr,
                   const mlir::RankedTensorType outType,
                   const llvm::ArrayRef<int64_t> multiples) {
    std::vector<float> inValues;
    if (!extractFloatValues(inputAttr, inValues)) return std::nullopt;

    const auto inType = llvm::cast<mlir::RankedTensorType>(inputAttr.getType());
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    if (multiples.size() != inShape.size() || outShape.size() != inShape.size()) return std::nullopt;

    for (size_t i = 0; i < inShape.size(); ++i) {
        if (multiples[i] <= 0) return std::nullopt;
        if (outShape[i] != inShape[i] * multiples[i]) return std::nullopt;
    }

    std::vector<float> outValues;
    outValues.reserve(outType.getNumElements());

    for (int64_t outFlat = 0, e = outType.getNumElements(); outFlat < e; ++outFlat) {
        const auto outIdx = unflattenIndex(outFlat, outShape);
        llvm::SmallVector<int64_t> inIdx(inShape.size(), 0);
        for (size_t i = 0; i < inShape.size(); ++i) {
            inIdx[i] = outIdx[i] % inShape[i];
        }
        const int64_t inFlat = flattenIndex(inIdx, inShape);
        outValues.push_back(inValues[inFlat]);
    }

    return createDenseElementsAttr(outValues, outType);
}

static std::optional<llvm::SmallVector<int64_t>> getConstI64Vector(const mlir::Value value) {
    mlir::Operation * const defOp = value.getDefiningOp();
    if (!defOp) return std::nullopt;

    const mlir::ElementsAttr tensorAttr = getConstTensorAttr(defOp);
    if (!tensorAttr) return std::nullopt;

    return extractI64Values(tensorAttr);
}

static std::optional<ResolvedConstTensor> resolveFloatConstTensor(const mlir::Value value) {
    mlir::Operation * const defOp = value.getDefiningOp();
    if (!defOp) return std::nullopt;

    if (const mlir::ElementsAttr tensorAttr = getConstTensorAttr(defOp)) {
        if (!mlir::isa<mlir::FloatType>(tensorAttr.getElementType()))
            return std::nullopt;

        return ResolvedConstTensor{ defOp, tensorAttr, tensorAttr };
    }

    const auto outType = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!outType || !mlir::isa<mlir::FloatType>(outType.getElementType())) return std::nullopt;

    if (llvm::isa<mlir::tosa::ReshapeOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto reshaped = reshapeFloatDenseAttr(inputResolved->resolvedTensorAttr, outType);
        if (!reshaped) return std::nullopt;

        inputResolved->resolvedTensorAttr = *reshaped;
        return inputResolved;
    }

    if (llvm::isa<mlir::tosa::CastOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto casted = castFloatDenseAttr(inputResolved->resolvedTensorAttr, outType);
        if (!casted) return std::nullopt;

        inputResolved->resolvedTensorAttr = *casted;
        return inputResolved;
    }

    if (llvm::isa<mlir::tosa::TransposeOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto perm = getConstI64Vector(defOp->getOperand(1));
        if (!perm) return std::nullopt;

        const auto transposed = transposeFloatDenseAttr(inputResolved->resolvedTensorAttr, outType, *perm);
        if (!transposed) return std::nullopt;

        inputResolved->resolvedTensorAttr = *transposed;
        return inputResolved;
    }

    if (llvm::isa<mlir::tosa::ReverseOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto axis = getI64ScalarAttr(defOp, {"axis"});
        if (!axis) return std::nullopt;

        const auto reversed = reverseFloatDenseAttr(inputResolved->resolvedTensorAttr, outType, *axis);
        if (!reversed) return std::nullopt;

        inputResolved->resolvedTensorAttr = *reversed;
        return inputResolved;
    }

    if (llvm::isa<mlir::tosa::SliceOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto start = getI64ArrayAttr(defOp, {"start"});
        const auto size = getI64ArrayAttr(defOp, {"size"});
        if (!start || !size) return std::nullopt;

        const auto sliced = sliceFloatDenseAttr(inputResolved->resolvedTensorAttr, outType, *start, *size);
        if (!sliced) return std::nullopt;

        inputResolved->resolvedTensorAttr = *sliced;
        return inputResolved;
    }

    if (llvm::isa<mlir::tosa::TileOp>(defOp)) {
        auto inputResolved = resolveFloatConstTensor(defOp->getOperand(0));
        if (!inputResolved) return std::nullopt;

        const auto multiples = getI64ArrayAttr(defOp, {"multiples"});
        if (!multiples) return std::nullopt;

        const auto tiled = tileFloatDenseAttr(inputResolved->resolvedTensorAttr, outType, *multiples);
        if (!tiled) return std::nullopt;

        inputResolved->resolvedTensorAttr = *tiled;
        return inputResolved;
    }

    return std::nullopt;
}

} // namespace

conquer::WeightPartitionKind
conquer::WeightAnalysisPass::getPartitionKind(mlir::Operation *consumerOp, const unsigned operandIndex) {
    if (llvm::isa<mlir::tosa::Conv2DOp>(consumerOp) && operandIndex == 1) return WeightPartitionKind::Axis;
    if (llvm::isa<mlir::tosa::DepthwiseConv2DOp>(consumerOp) && operandIndex == 1) return WeightPartitionKind::DepthwiseCM;
    if (llvm::isa<mlir::tosa::MatMulOp>(consumerOp) && operandIndex == 1) return WeightPartitionKind::Axis;

    return WeightPartitionKind::None;
}

int conquer::WeightAnalysisPass::getPerChannelAxis(mlir::Operation *consumerOp, const unsigned operandIndex) {
    if (llvm::isa<mlir::tosa::Conv2DOp>(consumerOp) && operandIndex == 1) return 0;
    if (llvm::isa<mlir::tosa::MatMulOp>(consumerOp) && operandIndex == 1) return 2;

    return -1;
}

void conquer::WeightAnalysisPass::runOnOperation() {
    L_INFO("Running Weight Analysis Pass.");
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(&getContext());

    std::vector<ConstStatsTask> constTasks;
    std::vector<WeightUseTask> useTasks;
    llvm::DenseSet<mlir::Operation *> seenRootConsts;

    module.walk([&](mlir::Operation *op) {
        if (llvm::isa<mlir::ModuleOp>(op) || op->hasTrait<mlir::OpTrait::IsTerminator>())
            return;

        // Annotate *ALL* float constants generally
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
            if (const mlir::ElementsAttr attr = getConstTensorAttr(op)) {
                if (mlir::isa<mlir::FloatType>(attr.getElementType())) {
                    if (seenRootConsts.insert(op).second) {
                        constTasks.push_back(ConstStatsTask{op, attr, CalibrationStats{}, SensitivityStats{}});
                    }
                }
            }
        }

        if (!isQuantisable(op))
            return;

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            const mlir::Value operand = op->getOperand(i);
            const auto resolved = resolveFloatConstTensor(operand);

            // If the operand traces back to a valid constant, it is a weight!
            if (!resolved)
                continue;

            // Failsafe: Ensure we captured the root of this consumed weight chain
            if (seenRootConsts.insert(resolved->rootConstOp).second) {
                constTasks.push_back(ConstStatsTask{
                    resolved->rootConstOp,
                    resolved->rootTensorAttr,
                    CalibrationStats{},
                    SensitivityStats{}
                });
            }

            const WeightPartitionKind partitionKind = getPartitionKind(op, i);
            const int axis = partitionKind == WeightPartitionKind::Axis ? getPerChannelAxis(op, i) : -1;

            useTasks.push_back(WeightUseTask{
                resolved->rootConstOp,
                op,
                i,
                resolved->resolvedTensorAttr,
                partitionKind,
                axis,
                CalibrationStats{},
                {},
                SensitivityStats{}
            });
        }
    });

    llvm::parallelForEach(constTasks, [&](ConstStatsTask &task) { analyseConstTask(task); });
    llvm::parallelForEach(useTasks, [&](WeightUseTask &task) { analyseUseTask(task); });

    L_DEBUG("Weight analysis complete. Analysed " << constTasks.size() << " root constants and " << useTasks.size() << " weight uses.");

    for (const auto &task : constTasks) { applyConstStatsToIR(task, builder); }
    for (const auto &task : useTasks) { applyUseStatsToIR(task, builder); }
}

void conquer::WeightAnalysisPass::analyseConstTask(ConstStatsTask &task) {
    if (!task.tensorAttr) return;
    L_TRACE("Analysing constant: " << compact(task.defOp));

    std::vector<float> values;
    if (!extractFloatValues(task.tensorAttr, values)) return;
    if (values.empty()) return;

    CalibrationCollector globalCollector;
    globalCollector.processBulk(values);
    task.globalStats = globalCollector.getStats();

    const auto shape = llvm::cast<mlir::ShapedType>(task.tensorAttr.getType()).getShape();
    if (!shape.empty()) {
        int64_t c_dim = shape.back();
        SensitivityCollector sensCollector;
        sensCollector.init(c_dim);
        sensCollector.updateStreaming(values, c_dim);
        task.globalSensitivity = sensCollector.getStats();
    }
}

void conquer::WeightAnalysisPass::analyseUseTask(WeightUseTask &task) {
    if (!task.tensorAttr) return;
    L_TRACE("Analysing weight use at " << compact(task.consumerOp) << " operand " << task.operandIndex);

    const auto shape = llvm::cast<mlir::ShapedType>(task.tensorAttr.getType()).getShape();
    if (shape.empty()) return;

    std::vector<float> values;
    if (!extractFloatValues(task.tensorAttr, values)) return;
    if (values.empty()) return;

    {
        CalibrationCollector globalCollector;
        globalCollector.processBulk(values);
        task.globalStats = globalCollector.getStats();
    }

    {
        int64_t c_dim = shape.back();
        SensitivityCollector sensCollector;
        sensCollector.init(c_dim);
        sensCollector.updateStreaming(values, c_dim);
        task.globalSensitivity = sensCollector.getStats();
    }

    if (task.partitionKind == WeightPartitionKind::None) return;

    if (task.partitionKind == WeightPartitionKind::DepthwiseCM) {
        if (shape.size() != 4) return;

        const int64_t KH = shape[0], KW = shape[1], C = shape[2], M = shape[3];
        const int64_t numChannels = C * M;
        task.channelStats.resize(numChannels);

        std::vector<std::vector<float>> channels(numChannels);
        for (auto &ch : channels) ch.reserve(KH * KW);

        auto flatIndex = [=](const int64_t kh, const int64_t kw, const int64_t c, const int64_t m) -> size_t {
            return static_cast<size_t>((((kh * KW) + kw) * C + c) * M + m);
        };

        for (int64_t kh = 0; kh < KH; ++kh) {
            for (int64_t kw = 0; kw < KW; ++kw) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t m = 0; m < M; ++m) {
                        const int64_t outCh = c * M + m;
                        channels[outCh].push_back(values[flatIndex(kh, kw, c, m)]);
                    }
                }
            }
        }

        llvm::parallelFor(size_t{0}, static_cast<size_t>(numChannels), [&](const size_t i) {
            CalibrationCollector channelCollector;
            channelCollector.processBulk(channels[i]);
            task.channelStats[i] = channelCollector.getStats();
        });

        return;
    }

    if (task.partitionKind != WeightPartitionKind::Axis) return;
    if (task.axis < 0 || task.axis >= static_cast<int>(shape.size())) return;

    const int64_t numChannels = shape[task.axis];
    const int64_t elementsPerChannel = static_cast<int64_t>(values.size()) / numChannels;

    int64_t stride = 1;
    for (size_t d = shape.size() - 1; d > static_cast<size_t>(task.axis); --d) {
        stride *= shape[d];
    }

    std::vector<std::vector<float>> channels(numChannels);
    for (auto &c : channels) c.reserve(elementsPerChannel);

    for (size_t i = 0; i < values.size(); ++i) {
        const int64_t c = static_cast<int64_t>(i / stride) % numChannels;
        channels[c].push_back(values[i]);
    }

    task.channelStats.resize(numChannels);

    llvm::parallelFor(size_t{0}, static_cast<size_t>(numChannels), [&](const size_t i) {
        CalibrationCollector channelCollector;
        channelCollector.processBulk(channels[i]);
        task.channelStats[i] = channelCollector.getStats();
    });
}

void conquer::WeightAnalysisPass::applyConstStatsToIR(const ConstStatsTask &task, mlir::Builder &builder) {
    if (!task.defOp) return;
    if (task.globalStats.min == std::numeric_limits<float>::max()) return;

    task.defOp->setAttr("weight_stats.min_max.per_tensor.min", builder.getF32FloatAttr(task.globalStats.min));
    task.defOp->setAttr("weight_stats.min_max.per_tensor.max", builder.getF32FloatAttr(task.globalStats.max));
    task.defOp->setAttr("weight_stats.percentile.per_tensor.min", builder.getF32FloatAttr(task.globalStats.pctMin));
    task.defOp->setAttr("weight_stats.percentile.per_tensor.max", builder.getF32FloatAttr(task.globalStats.pctMax));
    task.defOp->setAttr("weight_stats.kl.int8.per_tensor.min", builder.getF32FloatAttr(task.globalStats.klInt8Min));
    task.defOp->setAttr("weight_stats.kl.int8.per_tensor.max", builder.getF32FloatAttr(task.globalStats.klInt8Max));
    task.defOp->setAttr("weight_stats.kl.int4.per_tensor.min", builder.getF32FloatAttr(task.globalStats.klInt4Min));
    task.defOp->setAttr("weight_stats.kl.int4.per_tensor.max", builder.getF32FloatAttr(task.globalStats.klInt4Max));

    task.defOp->setAttr("sensitivity_value.entropy.weight_const", builder.getF32FloatAttr(task.globalSensitivity.entropySensitivity));
}

void conquer::WeightAnalysisPass::applyUseStatsToIR(const WeightUseTask &task, mlir::Builder &builder) {
    if (!task.consumerOp) return;

    const std::string prefix = "weight_use." + std::to_string(task.operandIndex) + ".";

    task.consumerOp->setAttr(prefix + "partition_kind", builder.getStringAttr(partitionKindToString(task.partitionKind)));

    if (task.partitionKind == WeightPartitionKind::Axis) {
        task.consumerOp->setAttr(prefix + "axis", builder.getI64IntegerAttr(task.axis));
    } else {
        task.consumerOp->removeAttr(prefix + "axis");
    }

    if (task.globalStats.min != std::numeric_limits<float>::max()) {
        task.consumerOp->setAttr(prefix + "min_max.per_tensor.min", builder.getF32FloatAttr(task.globalStats.min));
        task.consumerOp->setAttr(prefix + "min_max.per_tensor.max", builder.getF32FloatAttr(task.globalStats.max));
        task.consumerOp->setAttr(prefix + "percentile.per_tensor.min", builder.getF32FloatAttr(task.globalStats.pctMin));
        task.consumerOp->setAttr(prefix + "percentile.per_tensor.max", builder.getF32FloatAttr(task.globalStats.pctMax));
        task.consumerOp->setAttr(prefix + "kl.int8.per_tensor.min", builder.getF32FloatAttr(task.globalStats.klInt8Min));
        task.consumerOp->setAttr(prefix + "kl.int8.per_tensor.max", builder.getF32FloatAttr(task.globalStats.klInt8Max));
        task.consumerOp->setAttr(prefix + "kl.int4.per_tensor.min", builder.getF32FloatAttr(task.globalStats.klInt4Min));
        task.consumerOp->setAttr(prefix + "kl.int4.per_tensor.max", builder.getF32FloatAttr(task.globalStats.klInt4Max));
    }

    if (task.channelStats.empty()) return;

    llvm::SmallVector<mlir::Attribute> cMinsAttr, cMaxsAttr;
    llvm::SmallVector<mlir::Attribute> cPctMinsAttr, cPctMaxsAttr;
    llvm::SmallVector<mlir::Attribute> cKlInt8MinsAttr, cKlInt8MaxsAttr;
    llvm::SmallVector<mlir::Attribute> cKlInt4MinsAttr, cKlInt4MaxsAttr;

    for (const auto &stats : task.channelStats) {
        cMinsAttr.push_back(builder.getF32FloatAttr(stats.min));
        cMaxsAttr.push_back(builder.getF32FloatAttr(stats.max));
        cPctMinsAttr.push_back(builder.getF32FloatAttr(stats.pctMin));
        cPctMaxsAttr.push_back(builder.getF32FloatAttr(stats.pctMax));
        cKlInt8MinsAttr.push_back(builder.getF32FloatAttr(stats.klInt8Min));
        cKlInt8MaxsAttr.push_back(builder.getF32FloatAttr(stats.klInt8Max));
        cKlInt4MinsAttr.push_back(builder.getF32FloatAttr(stats.klInt4Min));
        cKlInt4MaxsAttr.push_back(builder.getF32FloatAttr(stats.klInt4Max));
    }

    task.consumerOp->setAttr(prefix + "min_max.per_channel.mins", builder.getArrayAttr(cMinsAttr));
    task.consumerOp->setAttr(prefix + "min_max.per_channel.maxes", builder.getArrayAttr(cMaxsAttr));
    task.consumerOp->setAttr(prefix + "percentile.per_channel.mins", builder.getArrayAttr(cPctMinsAttr));
    task.consumerOp->setAttr(prefix + "percentile.per_channel.maxes", builder.getArrayAttr(cPctMaxsAttr));
    task.consumerOp->setAttr(prefix + "kl.int8.per_channel.mins", builder.getArrayAttr(cKlInt8MinsAttr));
    task.consumerOp->setAttr(prefix + "kl.int8.per_channel.maxes", builder.getArrayAttr(cKlInt8MaxsAttr));
    task.consumerOp->setAttr(prefix + "kl.int4.per_channel.mins", builder.getArrayAttr(cKlInt4MinsAttr));
    task.consumerOp->setAttr(prefix + "kl.int4.per_channel.maxes", builder.getArrayAttr(cKlInt4MaxsAttr));

    std::string sensName = "sensitivity_value.entropy.weight_" + std::to_string(task.operandIndex);
    task.consumerOp->setAttr(sensName, builder.getF32FloatAttr(task.globalSensitivity.entropySensitivity));
}