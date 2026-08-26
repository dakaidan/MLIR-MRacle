#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include <random>

namespace mlir {
namespace metamorphic {

// Built-in ARMv8-specific transforms (target "armv8"). Only sound on a weak
// memory model; included in getTransforms() only when the requested target
// is "armv8".
bool tryCommuteRelaxedReadWrite(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryCommuteRelaxedWriteWrite(func::FuncOp, RewriterBase &, std::mt19937 &);

} // namespace metamorphic
} // namespace mlir
