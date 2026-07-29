#pragma once

#include "conquer/quantisation/policy.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

namespace conquer {
inline std::string getOpName(mlir::Operation *op) {
    if (const auto nameAttr = op->getAttrOfType<mlir::StringAttr>("conquer.name")) {
        return nameAttr.getValue().str();
    }
    return "";
}

inline bool should_quantise(mlir::Operation *op, const QuantisationPolicy &policy) {
    return policy.has_config(getOpName(op));
}

inline mlir::Type to_mlir_type(const DataType type, mlir::MLIRContext &context) {
    switch (type) {
    case DataType::FP32:
        return mlir::Float32Type::get(&context);
    case DataType::FP16:
        return mlir::Float16Type::get(&context);
    case DataType::BF16:
        return mlir::BFloat16Type::get(&context);
    case DataType::FP8_E4M3:
        return mlir::Float8E4M3FNType::get(&context);
    case DataType::FP8_E5M2:
        return mlir::Float8E5M2Type::get(&context);
    case DataType::INT32:
        return mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless);
    case DataType::INT16:
        return mlir::IntegerType::get(&context, 16, mlir::IntegerType::Signless);
    case DataType::INT8:
        return mlir::IntegerType::get(&context, 8, mlir::IntegerType::Signless);
    case DataType::INT4:
        return mlir::IntegerType::get(&context, 8, mlir::IntegerType::Signless);
    default:
        throw std::runtime_error("Unsupported data type for MLIR conversion");
    }
}

struct LayerTarget {
    std::optional<mlir::Type> target_weight_type;
    std::optional<mlir::Type> target_activation_type;
};

inline LayerTarget get_target_type(mlir::MLIRContext &context, mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto configOpt = policy.get_config(getOpName(op))) {
        const auto &[weight_policy, activation_policy, _] = configOpt.value();
        LayerTarget target;

        if (weight_policy.has_value()) {
            target.target_weight_type = to_mlir_type(weight_policy.value().precision, context);
        }
        if (activation_policy.has_value()) {
            target.target_activation_type = to_mlir_type(activation_policy.value().precision, context);
        }

        return target;
    }
    throw std::runtime_error("No quantisation config found for operation: " + getOpName(op));
}

inline bool get_should_squash_acc(mlir::Operation *op, const QuantisationPolicy &policy) {
    if (const auto configOpt = policy.get_config(getOpName(op))) {
        if (configOpt.has_value()) {
            return configOpt.value().squash_acc;
        }
    }
    return false;
}

static int get_precision_rank(const mlir::Type type) {
    if (type.isF32()) return 10;
    if (type.isF16()) return 9;
    if (type.isBF16()) return 8;
    if (type.isF8E5M2()) return 7;
    if (type.isF8E4M3FN()) return 6;
    if (type.isInteger(32)) return 5;
    if (type.isInteger(16)) return 4;
    if (type.isInteger(8)) return 3;
    if (type.isInteger(4)) return 2;
    return 0;
}

inline mlir::Type get_highest_precision_type(const LayerTarget &target) {
    const int weight_rank = target.target_weight_type ? get_precision_rank(target.target_weight_type.value()) : 0;
    const int activation_rank = target.target_activation_type ? get_precision_rank(target.target_activation_type.value()) : 0;

    if (weight_rank == 0 && activation_rank == 0) return nullptr;

    if (weight_rank >= activation_rank) {
        return target.target_weight_type.value();
    }

    return target.target_activation_type.value();
}

inline bool is_activation(const mlir::Value val) {
    mlir::Value currentVal = val;

    while (true) {
        if (llvm::isa<mlir::BlockArgument>(currentVal)) {
            return true;
        }

        if (const auto definingOp = currentVal.getDefiningOp()) {
            if (definingOp->getNumOperands() > 0) {
                currentVal = definingOp->getOperand(0);
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
}

inline bool is_weight(const mlir::Value val) {
    mlir::Operation *defOp = val.getDefiningOp();

    if (!defOp) {
        return false;
    }

    if (!llvm::isa<mlir::tosa::ConstOp>(defOp)) {
        return false;
    }
    if (auto constOp = llvm::dyn_cast<mlir::tosa::ConstOp>(val.getDefiningOp())) {
        if (const auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(constOp.getType())) {
            if (!llvm::isa<mlir::FloatType>(tensorType.getElementType())) {
                return false;
            }
        }
    }

    for (const auto user : val.getUsers()) {
        if (llvm::isa<
            mlir::tosa::Conv2DOp,
            mlir::tosa::Conv3DOp,
            mlir::tosa::DepthwiseConv2DOp,
            mlir::tosa::TransposeConv2DOp,
            mlir::tosa::MatMulOp>(user)) {
            if (user->getNumOperands() > 1 && user->getOperand(1) == val) {
                return true;
            }
        }
        if (llvm::isa<
            mlir::tosa::AddOp,
            mlir::tosa::SubOp,
            mlir::tosa::MaximumOp,
            mlir::tosa::MinimumOp,
            mlir::tosa::MulOp,
            mlir::tosa::EqualOp,
            mlir::tosa::GreaterOp,
            mlir::tosa::GreaterEqualOp
            >(user)) {
            return user->getNumOperands() > 0 && (user->getOperand(0) == val || user->getOperand(1) == val);
        }
    }

    return false;
}

inline mlir::Type get_propagated_type(mlir::Operation *op) {
    const auto currentResultType = llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!currentResultType)
        return nullptr;

    const mlir::Type originalElemType = currentResultType.getElementType();
    mlir::Type targetElemType = originalElemType;

    const mlir::Value primaryInput = op->getOperand(0);
    if (auto castOp = primaryInput.getDefiningOp<mlir::tosa::CastOp>()) {
        if (const auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(castOp.getInput().getType())) {
            targetElemType = tensorType.getElementType();
        }
    }

    if (targetElemType == originalElemType) {
        llvm::SmallVector<mlir::tosa::CastOp, 4> castUsers;
        for (const auto user : op->getResult(0).getUsers()) {
            if (const auto castUser = llvm::dyn_cast<mlir::tosa::CastOp>(user)) {
                if (castUser->use_empty())
                    continue;
                castUsers.push_back(castUser);
            }
        }

        if (!castUsers.empty()) {
            mlir::Type highestElemType = llvm::cast<mlir::RankedTensorType>(castUsers[0].getType()).getElementType();
            for (size_t i = 1; i < castUsers.size(); ++i) {
                mlir::Type userElemType = llvm::cast<mlir::RankedTensorType>(castUsers[i].getType()).getElementType();
                LayerTarget mockTarget{userElemType, highestElemType};
                highestElemType = get_highest_precision_type(mockTarget);
            }
            targetElemType = highestElemType;
        }
    }
    return targetElemType;
}

inline bool check_precision(const mlir::Value val, const mlir::RankedTensorType current_type,
                            const std::optional<mlir::Type> target_type) {
    if (!target_type.has_value())
        return true;
    if (!current_type)
        return false;

    if (current_type.getElementType() == target_type.value())
        return true;

    if (auto castOp = val.getDefiningOp<mlir::tosa::CastOp>()) {
        if (const auto castInputType = llvm::dyn_cast<mlir::RankedTensorType>(castOp.getInput().getType())) {
            if (castInputType.getElementType() == target_type.value())
                return true;
        }
    }
    return false;
}
}