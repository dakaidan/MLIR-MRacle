#pragma once

#include "conquer/loaders/model/model.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <string>

namespace conquer {

class MLIRModelLoader : public ModelLoader {
public:
    mlir::OwningOpRef<mlir::Operation *>
    loadModel(const std::string &modelPath, mlir::MLIRContext *ctx) override;

    [[nodiscard]] bool supportsExtension(const std::string &extension) const override;

    [[nodiscard]] bool supportsFormat(const std::string &modelPath,
                                      mlir::MLIRContext *ctx) const override;
};
} // namespace conquer