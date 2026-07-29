#include "conquer/runtime/session.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/Tosa/Transforms/Passes.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-session"

void conquer::ModelSession::load(mlir::ModuleOp mod) {
    L_INFO("Loading model into session.");
    this->module = mod;
    expectedInputs.clear();
    expectedOutputs.clear();

    auto funcOp = mod.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp) {
        throw std::runtime_error("conquer::ModelSession: No 'main' function found.");
    }

    parseTypeList(funcOp.getArgumentTypes(), expectedInputs, "Input");
    parseTypeList(funcOp.getResultTypes(), expectedOutputs, "Output");

    L_DEBUG("Session loaded with " << expectedInputs.size() << " inputs and " << expectedOutputs.size() << " outputs.");
}

[[nodiscard]] bool conquer::ModelSession::validateInputs(const std::vector<TensorView> &inputs) const {
    return validateList(inputs, expectedInputs, "Input");
}

std::vector<std::vector<int64_t>> conquer::ModelSession::resolveOutputShapes(const std::vector<TensorView> &inputs) {
    bool allStatic = true;
    std::vector<std::vector<int64_t>> staticShapes;
    for (const auto &out : expectedOutputs) {
        if (!out.isStatic()) {
            allStatic = false;
            break;
        }
        staticShapes.push_back(out.dims);
    }

    if (allStatic) {
        return staticShapes;
    }

    L_DEBUG("Dynamic shapes detected, running shape inference.");
    return runShapeInference(inputs);
}

[[nodiscard]] conquer::OutputAllocation
conquer::ModelSession::allocateOutputs(const std::vector<std::vector<int64_t>> &shapes) const {
    if (shapes.size() != expectedOutputs.size()) {
        throw std::runtime_error("conquer::ModelSession: Shape count does not match expected output count.");
    }

    OutputAllocation allocation;
    allocation.buffers.reserve(shapes.size());
    allocation.views.reserve(shapes.size());

    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];
        const DataType dtype = expectedOutputs[i].expectedType;

        size_t numElements = 1;
        for (const auto d : shape)
            numElements *= d;

        const size_t elementSize = get_size_bytes(dtype);
        size_t totalBytes = numElements * elementSize;

        allocation.buffers.emplace_back(totalBytes);

        allocation.views.push_back({allocation.buffers.back().data(), shape, dtype});
    }

    return allocation;
}

std::vector<std::vector<int64_t>> conquer::ModelSession::runShapeInference(const std::vector<TensorView> &inputs) {
    auto moduleClone = module.clone();
    auto funcOp = moduleClone.lookupSymbol<mlir::func::FuncOp>("main");
    auto &entryBlock = funcOp.front();

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto arg = entryBlock.getArgument(i);
        if (auto oldType = llvm::dyn_cast<mlir::RankedTensorType>(arg.getType())) {
            const auto newType = mlir::RankedTensorType::get(inputs[i].shape, oldType.getElementType());
            arg.setType(newType);
        }
    }

    mlir::PassManager pm(moduleClone.getContext());
    pm.addPass(mlir::tosa::createTosaInferShapesPass());

    if (mlir::failed(pm.run(moduleClone))) {
        moduleClone->erase();
        throw std::runtime_error("conquer::ModelSession: Shape inference failed.");
    }

    std::vector<std::vector<int64_t>> calculatedShapes;

    for (const auto returnOp = funcOp.getBody().back().getTerminator(); auto operand : returnOp->getOperands()) {
        auto type = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
        calculatedShapes.push_back(type.getShape().vec());
    }

    moduleClone->erase();
    return calculatedShapes;
}

void conquer::ModelSession::parseTypeList(const mlir::TypeRange types, std::vector<TensorConstraint> &constraints,
                                          const std::string &prefix) {
    for (size_t i = 0; i < types.size(); ++i) {
        auto type = types[i];
        TensorConstraint c;
        c.debugName = prefix + "_" + std::to_string(i);

        if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
            c.rank = tensorType.getRank();
            for (auto shape = tensorType.getShape(); auto dim : shape) {
                c.dims.push_back(dim == mlir::ShapedType::kDynamic ? -1 : dim);
            }
            c.expectedType = mapMlirType(tensorType.getElementType());
        } else if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type)) {
            c.rank = memrefType.getRank();
            for (auto shape = memrefType.getShape(); auto dim : shape) {
                c.dims.push_back(dim == mlir::ShapedType::kDynamic ? -1 : dim);
            }
            c.expectedType = mapMlirType(memrefType.getElementType());
        } else {
            c.rank = -1;
        }

        if (c.rank != -1)
            constraints.push_back(c);
    }
}

bool conquer::ModelSession::validateList(const std::vector<TensorView> &views,
                                         const std::vector<TensorConstraint> &constraints, const std::string &context) {
    if (views.size() != constraints.size()) {
        L_DEBUG("[Session Error] " << context << " count mismatch. Expected " << constraints.size() << ", got " << views.size());
        return false;
    }

    for (size_t i = 0; i < views.size(); ++i) {
        const auto &view = views[i];
        const auto &constraint = constraints[i];

        if (view.dtype != constraint.expectedType) {
            L_DEBUG("[Session Error] " << context << " " << i << " type mismatch. Expected " << to_string(constraint.expectedType) << ", got " << to_string(view.dtype));
            return false;
        }
        if (static_cast<int64_t>(view.shape.size()) != constraint.rank) {
            L_DEBUG("[Session Error] " << context << " " << i << " rank mismatch. Expected " << constraint.rank << ", got " << view.shape.size());
            return false;
        }
        for (size_t d = 0; d < static_cast<size_t>(constraint.rank); ++d) {
            const int64_t expectedDim = constraint.dims[d];
            if (const int64_t actualDim = view.shape[d]; expectedDim != -1 && expectedDim != actualDim) {
                L_DEBUG("[Session Error] " << context << " " << i << " dim " << d << " mismatch. Expected " << expectedDim << ", got " << actualDim);
                return false;
            }
        }
    }
    return true;
}

conquer::DataType conquer::ModelSession::mapMlirType(const mlir::Type type) {
    if (type.isF32())
        return DataType::FP32;
    if (type.isF16())
        return DataType::FP16;
    if (type.isBF16())
        return DataType::BF16;
    if (type.isF8E5M2())
        return DataType::FP8_E5M2;
    if (type.isF8E4M3FN())
        return DataType::FP8_E4M3;
    if (type.isSignlessInteger(32))
        return DataType::INT32;
    if (type.isSignlessInteger(16))
        return DataType::INT16;
    // if (type.isUnsignedInteger(8))
    //     return DataType::UINT8;
    if (type.isSignlessInteger(8))
        return DataType::INT8;
    // if (type.isUnsignedInteger(4))
    //     return DataType::UINT4;
    if (type.isSignlessInteger(4))
        return DataType::INT4;
    return DataType::Unknown;
}
