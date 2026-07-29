#pragma once

#include "conquer/core/config.h"

#include <mlir/IR/Types.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/Support/Casting.h>
#include <optional>

namespace conquer {

enum class QuantMode { Float, Integer };

struct OpPrecisionPlan {
    mlir::Type operandStorageType;
    mlir::Type operandExecutionType;
    mlir::Type resultType;

    std::optional<mlir::Type> accumulatorType;
    std::optional<mlir::Type> biasType;
    std::optional<mlir::Type> inputzp;
    std::optional<mlir::Type> weightzp;
    std::optional<mlir::Type> outputzp;

    bool requiresRescale;
};

template <typename TosaOp, QuantMode Mode, typename TargetElementType>
struct TosaQuantTraits {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type specificTargetType = {}, bool compressAccumulator = false) {
        return std::nullopt;
    }
    static std::optional<OpPrecisionPlan> getMixedPlan(mlir::MLIRContext* ctx, mlir::Type inputType = {}, mlir::Type weightType = {}, bool compressAccumulator = false) {
        return std::nullopt;
    }
};

// ==============================================================================
// FLOAT DOT-PRODUCT OPS (CONV FAMILY)
// ==============================================================================
// Ops: Conv2DOp, Conv3DOp, DepthwiseConv2DOp, TransposeConv2DOp
// Rationale: TOSA explicitly permits higher precision accumulators for fp16 and bf16.
// FP8 types strictly require fp16 accumulators and bias.
//
// | Input   | Output | Default Acc | Compressed Acc | Bias | ZP   | Rescale |
// |---------|--------|-------------|----------------|------|------|---------|
// | fp32    | fp32   | fp32        | fp32           | fp32 | none | false   |
// | fp16    | fp16   | fp32        | fp16           | fp16 | none | false   |
// | bf16    | bf16   | fp32        | fp32 (strict)  | bf16 | none | false   |
// | fp8e5m2 | fp16   | fp16        | fp16           | fp16 | none | false   |
// | fp8e4m3 | fp16   | fp16        | fp16           | fp16 | none | false   |
// ==============================================================================
#define DECLARE_TOSA_FLOAT_CONV_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float32Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, t, mlir::Float32Type::get(ctx), t, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            const mlir::Type acc = compressAccumulator ? t : mlir::Float32Type::get(ctx); \
            return OpPrecisionPlan{t, t, t, acc, t, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::BFloat16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            const mlir::Type f32 = mlir::Float32Type::get(ctx); \
            return OpPrecisionPlan{t, t, t, f32, t, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float8E5M2Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            mlir::Type f16 = mlir::Float16Type::get(ctx); \
            return OpPrecisionPlan{t, t, f16, f16, f16, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float8E4M3FNType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            mlir::Type f16 = mlir::Float16Type::get(ctx); \
            return OpPrecisionPlan{t, t, f16, f16, f16, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    };

DECLARE_TOSA_FLOAT_CONV_TRAITS(mlir::tosa::Conv2DOp)
DECLARE_TOSA_FLOAT_CONV_TRAITS(mlir::tosa::Conv3DOp)
DECLARE_TOSA_FLOAT_CONV_TRAITS(mlir::tosa::DepthwiseConv2DOp)
DECLARE_TOSA_FLOAT_CONV_TRAITS(mlir::tosa::TransposeConv2DOp)

// ==============================================================================
// FLOAT MATMUL
// ==============================================================================
// Ops: MatMulOp
// Rationale: MatMul has no bias or ZP in float TOSA. Outputs are promoted for bf16,
// and optionally for fp16 based on compression.
//
// | Input   | Default Out | Compressed Out | Acc  | Bias | ZP      | Rescale |
// |---------|-------------|----------------|------|------|---------|---------|
// | fp32    | fp32        | fp32           | none | none | fp32    | false   |
// | fp16    | fp32        | fp16           | none | none | fp16    | false   |
// | bf16    | fp32        | fp32 (strict)  | none | none | bf16    | false   |
// | fp8e5m2 | fp16        | fp16           | none | none | fp8e5m2 | false   |
// | fp8e4m3 | fp16        | fp16           | none | none | fp8e4m3 | false   |
// ==============================================================================
template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Float, mlir::Float32Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        return OpPrecisionPlan{t, t, t, std::nullopt, std::nullopt, t, t, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Float, mlir::Float16Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        const mlir::Type res = compressAccumulator ? t : mlir::Float32Type::get(ctx);
        return OpPrecisionPlan{t, t, res, std::nullopt, std::nullopt, t, t, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Float, mlir::BFloat16Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        const mlir::Type f32 = mlir::Float32Type::get(ctx);
        return OpPrecisionPlan{t, t, f32, std::nullopt, std::nullopt, t, t, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Float, mlir::Float8E5M2Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        const mlir::Type f16 = mlir::Float16Type::get(ctx);
        return OpPrecisionPlan{t, t, f16, std::nullopt, std::nullopt, t, t, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Float, mlir::Float8E4M3FNType> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        const mlir::Type f16 = mlir::Float16Type::get(ctx);
        return OpPrecisionPlan{t, t, f16, std::nullopt, std::nullopt, t, t,  std::nullopt, false};
    }
};

// ==============================================================================
// FLOAT AVG_POOL2D
// ==============================================================================
// Ops: AvgPool2dOp
// Rationale: Accumulator is stored via attribute. FP8 requires fp16 accumulation.
//
// | Input   | Output | Default Acc | Compressed Acc | Bias | ZP   | Rescale |
// |---------|--------|-------------|----------------|------|------|---------|
// | fp32    | fp32   | fp32        | fp32           | none | none | false   |
// | fp16    | fp16   | fp32        | fp16           | none | none | false   |
// | bf16    | bf16   | fp32        | fp32 (strict)  | none | none | false   |
// | fp8e5m2 | fp16   | fp16        | fp16           | none | none | false   |
// | fp8e4m3 | fp16   | fp16        | fp16           | none | none | false   |
// ==============================================================================
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Float, mlir::Float32Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        return OpPrecisionPlan{t, t, t, t, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Float, mlir::Float16Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        mlir::Type acc = compressAccumulator ? t : mlir::Float32Type::get(ctx);
        return OpPrecisionPlan{t, t, t, acc, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Float, mlir::BFloat16Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        return OpPrecisionPlan{t, t, t, mlir::Float32Type::get(ctx), std::nullopt, std::nullopt, std::nullopt, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Float, mlir::Float8E5M2Type> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        return OpPrecisionPlan{t, t, t, mlir::Float16Type::get(ctx), std::nullopt, std::nullopt, std::nullopt, std::nullopt, false};
    }
};
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Float, mlir::Float8E4M3FNType> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) {
        return OpPrecisionPlan{t, t, t, mlir::Float16Type::get(ctx), std::nullopt, std::nullopt, std::nullopt, std::nullopt, false};
    }
};

// ==============================================================================
// FLOAT ELEMENTWISE & REDUCTIONS
// ==============================================================================
// Ops: Add, Sub, Mul, Maximum, Minimum, Pow, ReduceSum, ReduceProduct, ReduceMax, ReduceMin
// Rationale: FP8 is strictly banned for these ops. Only fp32, fp16, bf16 permitted.
//
// | Input   | Output  | Acc  | Bias | ZP   | Rescale |
// |---------|---------|------|------|------|---------|
// | fp32    | fp32    | none | none | none | false   |
// | fp16    | fp16    | none | none | none | false   |
// | bf16    | bf16    | none | none | none | false   |
// ==============================================================================
#define DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float32Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, t, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, t, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::BFloat16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, t, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    };

DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::AddOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::SubOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::MulOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::MaximumOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::MinimumOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::PowOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::ReduceSumOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::ReduceProductOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::ReduceMaxOp)
DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS(mlir::tosa::ReduceMinOp)

// ==============================================================================
// FLOAT COMPARISONS
// ==============================================================================
// Ops: Equal, Greater, GreaterEqual
// Rationale: Result is always i1. FP8 banned.
//
// | Input   | Output  | Acc  | Bias | ZP   | Rescale |
// |---------|---------|------|------|------|---------|
// | fp32    | i1      | none | none | none | false   |
// | fp16    | i1      | none | none | none | false   |
// | bf16    | i1      | none | none | none | false   |
// ==============================================================================
#define DECLARE_TOSA_FLOAT_COMPARISON_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float32Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 1), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 1), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::BFloat16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 1), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    };

DECLARE_TOSA_FLOAT_COMPARISON_TRAITS(mlir::tosa::EqualOp)
DECLARE_TOSA_FLOAT_COMPARISON_TRAITS(mlir::tosa::GreaterOp)
DECLARE_TOSA_FLOAT_COMPARISON_TRAITS(mlir::tosa::GreaterEqualOp)

// ==============================================================================
// FLOAT ARGMAX
// ==============================================================================
// Ops: ArgMaxOp
// Rationale: Result is always i32. FP8 is permitted for ArgMax in TOSA.
//
// | Input   | Output | Acc  | Bias | ZP   | Rescale |
// |---------|--------|------|------|------|---------|
// | fp32    | i32    | none | none | none | false   |
// | fp16    | i32    | none | none | none | false   |
// | bf16    | i32    | none | none | none | false   |
// | fp8e5m2 | i32    | none | none | none | false   |
// | fp8e4m3 | i32    | none | none | none | false   |
// ==============================================================================
#define DECLARE_TOSA_FLOAT_ARGMAX_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float32Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::BFloat16Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float8E5M2Type> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    }; \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Float, mlir::Float8E4M3FNType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            return OpPrecisionPlan{t, t, mlir::IntegerType::get(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    };

DECLARE_TOSA_FLOAT_ARGMAX_TRAITS(mlir::tosa::ArgMaxOp)

// ==============================================================================
// FFT STRICT FP32 EXCEPTION
// ==============================================================================
// Ops: FFT2dOp, RFFT2dOp
// Rationale: TOSA specifically mandates that these ONLY execute in fp32.
// ==============================================================================
#define DECLARE_TOSA_STRICT_FP32_TRAITS(OP_TYPE) \
    template <typename FloatMlirType> \
    struct TosaQuantTraits<OP_TYPE, QuantMode::Float, FloatMlirType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext* ctx, mlir::Type t, bool compressAccumulator) { \
            mlir::Type f32 = mlir::Float32Type::get(ctx); \
            return OpPrecisionPlan{t, f32, f32, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false}; \
        } \
    };

DECLARE_TOSA_STRICT_FP32_TRAITS(mlir::tosa::FFT2dOp)
DECLARE_TOSA_STRICT_FP32_TRAITS(mlir::tosa::RFFT2dOp)

// ==============================================================================
// INTEGER HELPERS
// ==============================================================================

inline std::optional<unsigned> getIntegerWidth(mlir::Type t) {
    if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(t)) {
        return intTy.getWidth();
    }
    return std::nullopt;
}

inline bool isIntWidth(const mlir::Type t, const unsigned width) {
    const auto w = getIntegerWidth(t);
    return w.has_value() && *w == width;
}

inline mlir::Type tosaInt(mlir::MLIRContext *ctx, const unsigned width) {
    return mlir::IntegerType::get(ctx, width);
}

// Utility to find the next viable fallback width (snap cascade)
inline std::optional<mlir::Type> getNextIntFallback(mlir::MLIRContext* ctx, mlir::Type t) {
    auto w = getIntegerWidth(t);
    if (!w) return std::nullopt;
    if (*w == 4) return tosaInt(ctx, 8);
    if (*w == 8) return tosaInt(ctx, 16);
    if (*w == 16) return tosaInt(ctx, 32);
    if (*w == 32) return tosaInt(ctx, 16);
    return std::nullopt;
}

// ==============================================================================
// INTEGER DOT-PRODUCT OPS (CONV FAMILY & MATMUL)
// ==============================================================================
// Ops: Conv2D, Conv3D, DepthwiseConv2D, TransposeConv2D, MatMul
//
// | Input   | Weight | Output | Acc   | Bias  | Input ZP | Weight ZP | Output ZP | Rescale |
// |---------|--------|--------|-------|-------|----------|-----------|-----------|---------|
// | int8    | int8   | int32  | int32 | int32 | int8     | int8      | none      | true    |
// | int8    | int4   | int32  | int32 | int32 | int8     | int4      | none      | true    |
// | int16   | int8   | int48  | int48 | int48 | int16    | int8      | none      | true    |
// | int16   | int16  | int48  | none  | none  | int16    | int16     | none      | true    | (MatMul Only)
// ==============================================================================
#define DECLARE_TOSA_INT_CONV_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Integer, mlir::IntegerType> { \
        static std::optional<OpPrecisionPlan> getMixedPlan(mlir::MLIRContext* ctx, mlir::Type inputType, mlir::Type weightType, bool /*compressAccumulator*/) { \
            if (isIntWidth(inputType, 8) && isIntWidth(weightType, 8)) { \
                mlir::Type i32 = tosaInt(ctx, 32); \
                return OpPrecisionPlan{inputType, inputType, i32, i32, i32, inputType, weightType, std::nullopt, true}; \
            } \
            if (isIntWidth(inputType, 8) && isIntWidth(weightType, 4) && conquer::IS_I4_WEIGHT_COMPUTE_SUPPORTED) { \
                mlir::Type i32 = tosaInt(ctx, 32); \
                return OpPrecisionPlan{inputType, inputType, i32, i32, i32, inputType, weightType, std::nullopt, true}; \
            } \
            if (isIntWidth(inputType, 16) && isIntWidth(weightType, 8) && conquer::IS_I16_ACTIVATION_SUPPORTED) { \
                mlir::Type i48 = tosaInt(ctx, 48); \
                return OpPrecisionPlan{inputType, inputType, i48, i48, i48, inputType, weightType, std::nullopt, true}; \
            } \
            return std::nullopt; \
        } \
    };

DECLARE_TOSA_INT_CONV_TRAITS(mlir::tosa::Conv2DOp)
DECLARE_TOSA_INT_CONV_TRAITS(mlir::tosa::Conv3DOp)
DECLARE_TOSA_INT_CONV_TRAITS(mlir::tosa::DepthwiseConv2DOp)
DECLARE_TOSA_INT_CONV_TRAITS(mlir::tosa::TransposeConv2DOp)

template <> struct TosaQuantTraits<mlir::tosa::MatMulOp, QuantMode::Integer, mlir::IntegerType> {
    static std::optional<OpPrecisionPlan> getMixedPlan(mlir::MLIRContext* ctx, mlir::Type inputType, mlir::Type weightType, bool /*compressAccumulator*/) {
        if (isIntWidth(inputType, 8) && isIntWidth(weightType, 8)) {
            const mlir::Type i32 = tosaInt(ctx, 32);
            return OpPrecisionPlan{inputType, inputType, i32, std::nullopt, std::nullopt, inputType, weightType, std::nullopt, true};
        }
        if (isIntWidth(inputType, 16) && isIntWidth(weightType, 16)) {
            const mlir::Type i48 = tosaInt(ctx, 48);
            return OpPrecisionPlan{inputType, inputType, i48, std::nullopt, std::nullopt, inputType, weightType, std::nullopt, true};
        }
        return std::nullopt;
    }
};

// ==============================================================================
// INTEGER COMMON-DOMAIN OPS
// ==============================================================================
// Ops: Add, Sub, Maximum, Minimum, ReduceSum
// Rationale: Computed natively in i32 domain.
//
// | Input        | Exec | Output | Acc  | Bias | ZP   | Rescale |
// |--------------|------|--------|------|------|------|---------|
// | i8, i16, i32 | i32  | i32    | none | none | none | true    |
// ==============================================================================
#define DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Integer, mlir::IntegerType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) { \
            if (!isIntWidth(t, 8) && !isIntWidth(t, 16) && !isIntWidth(t, 32)) { \
                return std::nullopt; \
            } \
            mlir::Type i32 = tosaInt(ctx, 32); \
            return OpPrecisionPlan{ \
                t, i32, i32, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true \
            }; \
        } \
    };

DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(mlir::tosa::AddOp)
DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(mlir::tosa::SubOp)
DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(mlir::tosa::MaximumOp)
DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(mlir::tosa::MinimumOp)
DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS(mlir::tosa::ReduceSumOp)

// ==============================================================================
// INTEGER MUL
// ==============================================================================
// Ops: Mul
// Rationale: Multiplies two integers, producing an i32 result natively.
//
// | Input        | Exec   | Output | Acc  | Bias | ZP   | Rescale |
// |--------------|--------|--------|------|------|------|---------|
// | i8, i16, i32 | target | i32    | none | none | none | true    |
// ==============================================================================
#define DECLARE_I32_RESULT_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Integer, mlir::IntegerType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) { \
            if (!isIntWidth(t, 8) && !isIntWidth(t, 16) && !isIntWidth(t, 32)) { \
                return std::nullopt; \
            } \
            mlir::Type i32 = tosaInt(ctx, 32); \
            return OpPrecisionPlan{ \
                t, t, i32, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true \
            }; \
        } \
    };

DECLARE_I32_RESULT_TRAITS(mlir::tosa::MulOp)

// ==============================================================================
// INTEGER COMPARISONS
// ==============================================================================
// Ops: Equal, Greater, GreaterEqual
// Rationale: Executes in i32, returns i1.
//
// | Input        | Exec | Output | Acc  | Bias | ZP   | Rescale |
// |--------------|------|--------|------|------|------|---------|
// | i8, i16, i32 | i32  | i1     | none | none | none | true    |
// ==============================================================================
#define DECLARE_TOSA_INT_COMPARE_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Integer, mlir::IntegerType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) { \
            if (!isIntWidth(t, 8) && !isIntWidth(t, 16) && !isIntWidth(t, 32)) { \
                return std::nullopt; \
            } \
            mlir::Type i32 = tosaInt(ctx, 32); \
            mlir::Type i1  = tosaInt(ctx, 1); \
            return OpPrecisionPlan{ \
                t, i32, i1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true \
            }; \
        } \
    };

DECLARE_TOSA_INT_COMPARE_TRAITS(mlir::tosa::EqualOp)
DECLARE_TOSA_INT_COMPARE_TRAITS(mlir::tosa::GreaterOp)
DECLARE_TOSA_INT_COMPARE_TRAITS(mlir::tosa::GreaterEqualOp)

// ==============================================================================
// INTEGER REDUCE EXTREMA & ARITHMETIC RIGHT SHIFT
// ==============================================================================
// Ops: ReduceMax, ReduceMin, ArithmeticRightShift
// Rationale: Order is preserved or shifting stays within the same bitwidth.
//
// | Input        | Exec   | Output | Acc  | Bias | ZP   | Rescale |
// |--------------|--------|--------|------|------|------|---------|
// | i8, i16, i32 | target | target | none | none | none | false   |
// ==============================================================================
#define DECLARE_TOSA_INT_SAME_OUT_TRAITS(OP_TYPE) \
    template <> struct TosaQuantTraits<OP_TYPE, QuantMode::Integer, mlir::IntegerType> { \
        static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) { \
            if (!isIntWidth(t, 8) && !isIntWidth(t, 16) && !isIntWidth(t, 32)) { \
                return std::nullopt; \
            } \
            return OpPrecisionPlan{ \
                t, t, t, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false \
            }; \
        } \
    };

DECLARE_TOSA_INT_SAME_OUT_TRAITS(mlir::tosa::ReduceMaxOp)
DECLARE_TOSA_INT_SAME_OUT_TRAITS(mlir::tosa::ReduceMinOp)
DECLARE_TOSA_INT_SAME_OUT_TRAITS(mlir::tosa::ArithmeticRightShiftOp)

// ==============================================================================
// INTEGER ARGMAX
// ==============================================================================
// Ops: ArgMax
// Rationale: Returns i32 natively. Only i8 and i16 supported natively.
//
// | Input        | Exec   | Output | Acc  | Bias | ZP   | Rescale |
// |--------------|--------|--------|------|------|------|---------|
// | i8, i16      | target | i32    | none | none | none | false   |
// ==============================================================================
template <> struct TosaQuantTraits<mlir::tosa::ArgMaxOp, QuantMode::Integer, mlir::IntegerType> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) {
        if (!isIntWidth(t, 8) && !isIntWidth(t, 16)) return std::nullopt;
        return OpPrecisionPlan{
            t, t, tosaInt(ctx, 32), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false
        };
    }
};

// ==============================================================================
// INTEGER AVG_POOL2D
// ==============================================================================
// Ops: AvgPool2d
// Rationale: Requires i32 accumulator. Input/Output preserve type. Only i8/i16 natively.
//
// | Input   | Exec   | Output | Acc | Bias | Input ZP | Output ZP | Rescale |
// |---------|--------|--------|-----|------|----------|-----------|---------|
// | i8, i16 | target | target | i32 | none | target   | target    | false   |
// ==============================================================================
template <> struct TosaQuantTraits<mlir::tosa::AvgPool2dOp, QuantMode::Integer, mlir::IntegerType> {
    static std::optional<OpPrecisionPlan> getPlan(mlir::MLIRContext *ctx, mlir::Type t, bool /*compressAccumulator*/) {
        if (!isIntWidth(t, 8) && !isIntWidth(t, 16)) return std::nullopt;
        mlir::Type i32 = tosaInt(ctx, 32);
        return OpPrecisionPlan{
            t, t, t, i32, std::nullopt, t, std::nullopt, t, false
        };
    }
};

// ==============================================================================
// RUNTIME DISPATCHERS
// ==============================================================================
inline std::optional<OpPrecisionPlan> planTosaFloatMode(mlir::Operation* op, mlir::Type targetType, bool compressAccumulator = false) {
    mlir::MLIRContext* ctx = op->getContext();

    auto checkType = [&]<typename T0>([[maybe_unused]] T0 OpClass) -> std::optional<OpPrecisionPlan> {
        std::optional<OpPrecisionPlan> plan = std::nullopt;

        if (targetType.isF32()) plan = TosaQuantTraits<T0, QuantMode::Float, mlir::Float32Type>::getPlan(ctx, targetType, compressAccumulator);
        else if (targetType.isF16()) plan = TosaQuantTraits<T0, QuantMode::Float, mlir::Float16Type>::getPlan(ctx, targetType, compressAccumulator);
        else if (targetType.isBF16()) plan = TosaQuantTraits<T0, QuantMode::Float, mlir::BFloat16Type>::getPlan(ctx, targetType, compressAccumulator);
        else if (targetType.isF8E5M2()) plan = TosaQuantTraits<T0, QuantMode::Float, mlir::Float8E5M2Type>::getPlan(ctx, targetType, compressAccumulator);
        else if (targetType.isF8E4M3FN()) plan = TosaQuantTraits<T0, QuantMode::Float, mlir::Float8E4M3FNType>::getPlan(ctx, targetType, compressAccumulator);

        if (plan) return plan;

        if (targetType.isF8E5M2() || targetType.isF8E4M3FN()) {
            mlir::Type fallbackExecType = mlir::Float16Type::get(ctx);
            plan = TosaQuantTraits<T0, QuantMode::Float, mlir::Float16Type>::getPlan(ctx, fallbackExecType, compressAccumulator);
            if (plan) {
                plan->operandStorageType = targetType;
                return plan;
            }
        }
        return std::nullopt;
    };

    if (const auto convOp = llvm::dyn_cast<mlir::tosa::Conv2DOp>(op)) return checkType(convOp);
    if (const auto conv3dOp = llvm::dyn_cast<mlir::tosa::Conv3DOp>(op)) return checkType(conv3dOp);
    if (const auto dwConvOp = llvm::dyn_cast<mlir::tosa::DepthwiseConv2DOp>(op)) return checkType(dwConvOp);
    if (const auto tConvOp = llvm::dyn_cast<mlir::tosa::TransposeConv2DOp>(op)) return checkType(tConvOp);
    if (const auto matMulOp = llvm::dyn_cast<mlir::tosa::MatMulOp>(op)) return checkType(matMulOp);

    if (const auto addOp = llvm::dyn_cast<mlir::tosa::AddOp>(op)) return checkType(addOp);
    if (const auto subOp = llvm::dyn_cast<mlir::tosa::SubOp>(op)) return checkType(subOp);
    if (const auto mulOp = llvm::dyn_cast<mlir::tosa::MulOp>(op)) return checkType(mulOp);
    if (const auto maxOp = llvm::dyn_cast<mlir::tosa::MaximumOp>(op)) return checkType(maxOp);
    if (const auto minOp = llvm::dyn_cast<mlir::tosa::MinimumOp>(op)) return checkType(minOp);
    if (const auto powOp = llvm::dyn_cast<mlir::tosa::PowOp>(op)) return checkType(powOp);

    if (const auto arsOp = llvm::dyn_cast<mlir::tosa::ArithmeticRightShiftOp>(op)) return checkType(arsOp);

    if (const auto eqOp = llvm::dyn_cast<mlir::tosa::EqualOp>(op)) return checkType(eqOp);
    if (const auto grOp = llvm::dyn_cast<mlir::tosa::GreaterOp>(op)) return checkType(grOp);
    if (const auto geqOp = llvm::dyn_cast<mlir::tosa::GreaterEqualOp>(op)) return checkType(geqOp);

    if (const auto argMaxOp = llvm::dyn_cast<mlir::tosa::ArgMaxOp>(op)) return checkType(argMaxOp);

    if (const auto avgPoolOp = llvm::dyn_cast<mlir::tosa::AvgPool2dOp>(op)) return checkType(avgPoolOp);
    if (const auto rSumOp = llvm::dyn_cast<mlir::tosa::ReduceSumOp>(op)) return checkType(rSumOp);
    if (const auto rProdOp = llvm::dyn_cast<mlir::tosa::ReduceProductOp>(op)) return checkType(rProdOp);
    if (const auto rMaxOp = llvm::dyn_cast<mlir::tosa::ReduceMaxOp>(op)) return checkType(rMaxOp);
    if (const auto rMinOp = llvm::dyn_cast<mlir::tosa::ReduceMinOp>(op)) return checkType(rMinOp);

    if (const auto fftOp = llvm::dyn_cast<mlir::tosa::FFT2dOp>(op)) return checkType(fftOp);
    if (const auto rfftOp = llvm::dyn_cast<mlir::tosa::RFFT2dOp>(op)) return checkType(rfftOp);

    return std::nullopt;
}

inline std::optional<OpPrecisionPlan> planTosaIntMode(mlir::Operation* op, mlir::Type targetType, bool compressAccumulator = false) {
    if (!mlir::isa<mlir::IntegerType>(targetType)) return std::nullopt;
    mlir::MLIRContext* ctx = op->getContext();

    auto checkType = [&]<typename T0>([[maybe_unused]] T0 OpClass) -> std::optional<OpPrecisionPlan> {
        mlir::Type currentExecType = targetType;

        for (int attempt = 0; attempt < 4; ++attempt) {
            if (auto plan = TosaQuantTraits<T0, QuantMode::Integer, mlir::IntegerType>::getPlan(ctx, currentExecType, compressAccumulator)) {
                plan->operandStorageType = targetType;
                return plan;
            }
            auto fallbackOpt = getNextIntFallback(ctx, currentExecType);
            if (!fallbackOpt) break;
            currentExecType = *fallbackOpt;
        }
        return std::nullopt;
    };

    if (const auto addOp = llvm::dyn_cast<mlir::tosa::AddOp>(op)) return checkType(addOp);
    if (const auto subOp = llvm::dyn_cast<mlir::tosa::SubOp>(op)) return checkType(subOp);
    if (const auto mulOp = llvm::dyn_cast<mlir::tosa::MulOp>(op)) return checkType(mulOp);
    if (const auto maxOp = llvm::dyn_cast<mlir::tosa::MaximumOp>(op)) return checkType(maxOp);
    if (const auto minOp = llvm::dyn_cast<mlir::tosa::MinimumOp>(op)) return checkType(minOp);

    if (const auto arsOp = llvm::dyn_cast<mlir::tosa::ArithmeticRightShiftOp>(op)) return checkType(arsOp);
    if (const auto eqOp = llvm::dyn_cast<mlir::tosa::EqualOp>(op)) return checkType(eqOp);
    if (const auto grOp = llvm::dyn_cast<mlir::tosa::GreaterOp>(op)) return checkType(grOp);
    if (const auto geqOp = llvm::dyn_cast<mlir::tosa::GreaterEqualOp>(op)) return checkType(geqOp);

    if (const auto argMaxOp = llvm::dyn_cast<mlir::tosa::ArgMaxOp>(op)) return checkType(argMaxOp);
    if (const auto avgPoolOp = llvm::dyn_cast<mlir::tosa::AvgPool2dOp>(op)) return checkType(avgPoolOp);

    if (const auto rSumOp = llvm::dyn_cast<mlir::tosa::ReduceSumOp>(op)) return checkType(rSumOp);
    if (const auto rMaxOp = llvm::dyn_cast<mlir::tosa::ReduceMaxOp>(op)) return checkType(rMaxOp);
    if (const auto rMinOp = llvm::dyn_cast<mlir::tosa::ReduceMinOp>(op)) return checkType(rMinOp);

    return std::nullopt;
}

inline std::optional<OpPrecisionPlan> planTosaIntMixedMode(mlir::Operation* op, mlir::Type inputType, const mlir::Type weightType, bool compressAccumulator = false) {
    if (!mlir::isa<mlir::IntegerType>(inputType) || !mlir::isa<mlir::IntegerType>(weightType)) return std::nullopt;
    mlir::MLIRContext* ctx = op->getContext();

    auto checkMixedType = [&]<typename T0>([[maybe_unused]] T0 OpClass) -> std::optional<OpPrecisionPlan> {
        mlir::Type currIn = inputType;
        mlir::Type currWt = weightType;

        // Cascade search for nearest valid mixed configuration
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (auto plan = TosaQuantTraits<T0, QuantMode::Integer, mlir::IntegerType>::getMixedPlan(ctx, currIn, currWt, compressAccumulator)) {
                plan->operandStorageType = inputType;
                plan->inputzp = currIn;
                plan->weightzp = currWt;
                plan->operandExecutionType = currIn;
                return plan;
            }

            const unsigned iw = getIntegerWidth(currIn).value_or(0);
            const unsigned ww = getIntegerWidth(currWt).value_or(0);
            if (iw == 0 || ww == 0) break;

            // Promote the lowest precision parameter first
            if (iw < ww) {
                currIn = getNextIntFallback(ctx, currIn).value_or(currIn);
            } else if (ww < iw) {
                currWt = getNextIntFallback(ctx, currWt).value_or(currWt);
            } else {
                currIn = getNextIntFallback(ctx, currIn).value_or(currIn);
                currWt = getNextIntFallback(ctx, currWt).value_or(currWt);
            }
        }
        return std::nullopt;
    };

    if (const auto convOp = llvm::dyn_cast<mlir::tosa::Conv2DOp>(op)) return checkMixedType(convOp);
    if (const auto conv3dOp = llvm::dyn_cast<mlir::tosa::Conv3DOp>(op)) return checkMixedType(conv3dOp);
    if (const auto dwConvOp = llvm::dyn_cast<mlir::tosa::DepthwiseConv2DOp>(op)) return checkMixedType(dwConvOp);
    if (const auto tConvOp = llvm::dyn_cast<mlir::tosa::TransposeConv2DOp>(op)) return checkMixedType(tConvOp);
    if (const auto matMulOp = llvm::dyn_cast<mlir::tosa::MatMulOp>(op)) return checkMixedType(matMulOp);

    return std::nullopt;
}

// ==============================================================================
// ABSORB FLOAT / INT LEGALISATION
// ==============================================================================
struct AbsorbFloatPlan {
    mlir::Type opType;
    mlir::Type finalType;
    [[nodiscard]] bool requiresTrailingCast() const { return opType != finalType; }
};

struct AbsorbIntPlan {
    mlir::Type opType;
    mlir::Type finalType;
    [[nodiscard]] bool requiresTrailingCast() const { return opType != finalType; }
};

template <typename OpTy> struct TosaAbsorbFloatTraits { static constexpr bool supportsFp8 = false; };
template <typename OpTy> struct TosaAbsorbIntTraits { static constexpr bool supportsInt = true; };

// Safe absorb ops for both floats and integers
#define DECLARE_TOSA_ABSORB_TRAITS(OP_TYPE) \
    template <> struct TosaAbsorbFloatTraits<OP_TYPE> { static constexpr bool supportsFp8 = true; }; \
    template <> struct TosaAbsorbIntTraits<OP_TYPE> { static constexpr bool supportsInt = true; };

DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::ConcatOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::PadOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::ReshapeOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::ReverseOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::SliceOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::TileOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::TransposeOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::GatherOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::ScatterOp)
DECLARE_TOSA_ABSORB_TRAITS(mlir::tosa::MaxPool2dOp)

template <typename OpTy>
inline std::optional<AbsorbFloatPlan> planTosaAbsorbFloat(mlir::MLIRContext *ctx, const mlir::Type requestedType) {
    if (!mlir::isa<mlir::FloatType>(requestedType)) return std::nullopt;

    if (requestedType.isF32() || requestedType.isF16() || requestedType.isBF16()) {
        return AbsorbFloatPlan{requestedType, requestedType};
    }

    if (requestedType.isF8E5M2() || requestedType.isF8E4M3FN()) {
        if constexpr (TosaAbsorbFloatTraits<OpTy>::supportsFp8) {
            return AbsorbFloatPlan{requestedType, requestedType};
        }
        return AbsorbFloatPlan{mlir::Float16Type::get(ctx), requestedType};
    }
    return std::nullopt;
}

template <typename OpTy>
inline std::optional<AbsorbIntPlan> planTosaAbsorbInt(mlir::MLIRContext *ctx, const mlir::Type requestedType) {
    if (!mlir::isa<mlir::IntegerType>(requestedType)) return std::nullopt;

    // TOSA natively supports integer absorb ops across standardized widths (i8, i16, i32)
    // If the layout op strictly requires these, we snap up.
    if constexpr (TosaAbsorbIntTraits<OpTy>::supportsInt) {
        mlir::Type execType = requestedType;

        // Prevent layout ops from crashing out on pure i4 execution
        // by snapping directly up to i8.
        const unsigned w = getIntegerWidth(requestedType).value_or(0);
        if (w == 4) execType = tosaInt(ctx, 8);

        return AbsorbIntPlan{execType, requestedType};
    }
    return std::nullopt;
}

template <typename OpTy>
inline std::optional<OpPrecisionPlan> planRequestedFloatMode(mlir::Operation *op, const mlir::Type requestedType, const bool compressAccumulator = false) {
    if (const auto exact = planTosaFloatMode(op, requestedType, compressAccumulator)) return exact;

    if (requestedType.isF8E4M3FN() || requestedType.isF8E5M2()) {
        auto legalized = planTosaAbsorbFloat<OpTy>(op->getContext(), requestedType);
        if (!legalized) return std::nullopt;
        return planTosaFloatMode(op, legalized->opType, compressAccumulator);
    }
    return std::nullopt;
}

template <typename OpTy>
inline std::optional<OpPrecisionPlan> planRequestedIntMode(mlir::Operation *op, const mlir::Type requestedType, const bool compressAccumulator = false) {
    if (const auto exact = planTosaIntMode(op, requestedType, compressAccumulator)) return exact;

    // If it's an absorb op that doesn't inherently sit in the generic planner
    if (auto legalized = planTosaAbsorbInt<OpTy>(op->getContext(), requestedType)) {
        return planTosaIntMode(op, legalized->opType, compressAccumulator);
    }
    return std::nullopt;
}

template <typename OpTy>
inline std::optional<OpPrecisionPlan> planRequestedIntMixedMode(mlir::Operation *op, const mlir::Type inputType, const mlir::Type weightType, const bool compressAccumulator = false) {
    return planTosaIntMixedMode(op, inputType, weightType, compressAccumulator);
}

#undef DECLARE_TOSA_FLOAT_CONV_TRAITS
#undef DECLARE_TOSA_FLOAT_ELEMENTWISE_TRAITS
#undef DECLARE_TOSA_FLOAT_COMPARISON_TRAITS
#undef DECLARE_TOSA_FLOAT_ARGMAX_TRAITS
#undef DECLARE_TOSA_STRICT_FP32_TRAITS
#undef DECLARE_TOSA_INT_CONV_TRAITS
#undef DECLARE_TOSA_INT_COMMON_I32_RESULT_TRAITS
#undef DECLARE_I32_RESULT_TRAITS
#undef DECLARE_TOSA_INT_COMPARE_TRAITS
#undef DECLARE_TOSA_INT_SAME_OUT_TRAITS
#undef DECLARE_TOSA_ABSORB_TRAITS

} // namespace conquer