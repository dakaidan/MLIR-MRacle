#pragma once

#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace metamorphic {

// Built-in registry tables, merged into the shared registry by
// MetamorphicTransform.cpp.
llvm::ArrayRef<FunctionTransform> getGenericTransforms();
llvm::ArrayRef<FunctionTransform> getArmTransforms();

} // namespace metamorphic
} // namespace mlir
