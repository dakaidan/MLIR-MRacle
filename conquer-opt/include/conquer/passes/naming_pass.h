#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace conquer {
/// An MLIR pass that assigns a unique, human-readable name to each operation in a module.
/// The names are based on the operation's type (e.g., `tosa_conv2d_0`, `tosa_conv2d_1`, etc.).
/// This is a prerequisite for applying per-operation quantisation policies, as the operation
/// names are used as keys in the configuration file.
struct ModelNamingPass : public mlir::PassWrapper<ModelNamingPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModelNamingPass)
    void getDependentDialects(mlir::DialectRegistry &registry) const override {}

    /// Returns the command-line argument that can be used to run this pass.
    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-naming"; }
    /// Returns a short description of the pass.
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Assigns stable, unique names to all operations of models";
    }

    void runOnOperation() override;
};

inline std::unique_ptr<mlir::Pass> createModelNamingPass() { return std::make_unique<ModelNamingPass>(); }
} // namespace conquer
