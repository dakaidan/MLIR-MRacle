#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace conquer {
    /// An MLIR pass that finds large inline DenseElementsAttr constants
    /// and moves them into opaque DenseResourceElementsAttr blobs.
    struct ExternaliseConstantsPass : public mlir::PassWrapper<ExternaliseConstantsPass, mlir::OperationPass<mlir::ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternaliseConstantsPass)

        void getDependentDialects(mlir::DialectRegistry &registry) const override {}

        /// Returns the command-line argument that can be used to run this pass.
        [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-externalise-constants"; }

        /// Returns a short description of the pass.
        [[nodiscard]] llvm::StringRef getDescription() const override {
            return "Moves large inline constants into dialect resources to speed up compilation and I/O.";
        }

        void runOnOperation() override;
    };

    inline std::unique_ptr<mlir::Pass> createExternaliseConstantsPass() {
        return std::make_unique<ExternaliseConstantsPass>();
    }
} // namespace conquer