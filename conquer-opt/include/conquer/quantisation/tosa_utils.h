#pragma once

#include <mlir/IR/Operation.h>

namespace conquer {
bool isQuantisable(mlir::Operation *op);
}