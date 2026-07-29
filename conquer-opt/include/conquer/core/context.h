#pragma once
#include <mlir/IR/MLIRContext.h>

#include <memory>

namespace conquer {
    std::unique_ptr<mlir::MLIRContext> createMLIRContext();
} // namespace conquer