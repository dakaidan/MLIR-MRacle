#include "conquer/quantisation/tosa_utils.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-dialects"

bool conquer::isQuantisable(mlir::Operation *op) {
    if (op->getDialect()->getNamespace() != "tosa") {
        return false;
    }

    if (llvm::isa<// 1. Accumulating Tensor Ops (Math)
                  mlir::tosa::Conv2DOp,
                  mlir::tosa::Conv3DOp,
                  mlir::tosa::DepthwiseConv2DOp,
                  mlir::tosa::TransposeConv2DOp,
                  mlir::tosa::MatMulOp,

                  // 2. Elementwise Binary Math Ops (Requires Common Domain)
                  mlir::tosa::AddOp,
                  mlir::tosa::SubOp,
                  mlir::tosa::MulOp,
                  mlir::tosa::PowOp,
                  mlir::tosa::MaximumOp,
                  mlir::tosa::MinimumOp,
                  mlir::tosa::ArithmeticRightShiftOp,
                  mlir::tosa::EqualOp,
                  mlir::tosa::GreaterOp,
                  mlir::tosa::GreaterEqualOp,

                  // 3. Reduction Math Ops
                  mlir::tosa::AvgPool2dOp,
                  mlir::tosa::ReduceSumOp,
                  mlir::tosa::ReduceProductOp,
                  mlir::tosa::ReduceMaxOp,
                  mlir::tosa::ReduceMinOp//,

                  // // 4. Strict FP32 Exceptions
                  // mlir::tosa::FFT2dOp,
                  // mlir::tosa::RFFT2dOp
                  >(op)) {
        return true;
    }

    L_TRACE("Excluding op (passthrough or unsupported): " << compact(op));
    return false;
}