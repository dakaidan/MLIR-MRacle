#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace conquer {
struct ActivationInstrumentationPass
    : public mlir::PassWrapper<ActivationInstrumentationPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ActivationInstrumentationPass)
    void getDependentDialects(mlir::DialectRegistry &registry) const override {}

    /// Returns the command-line argument that can be used to run this pass.
    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-activation-instrumentation"; }
    /// Returns a short description of the pass.
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Instruments activation tensors to capture runtime statistics for quantisation calibration.";
    }

    void runOnOperation() override;
};

inline std::unique_ptr<mlir::Pass> createActivationInstrumentationPass() {
    return std::make_unique<ActivationInstrumentationPass>();
}
} // namespace conquer
