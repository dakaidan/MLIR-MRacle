#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace conquer {
/// An mlir pass that removes all annotations we added (naming, calibration, etc) to return to a clean state for export
/// or further processing.
struct StripAnnotationPass : public mlir::PassWrapper<StripAnnotationPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripAnnotationPass)

    void getDependentDialects(mlir::DialectRegistry &registry) const override {}

    /// Returns the command-line argument that can be used to run this pass.
    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-strip-annotations"; }
    /// Returns a short description of the pass.
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Removes all annotations (naming, calibration, etc) to return to a clean state for export or further "
               "processing.";
    }

    void runOnOperation() override;
};

inline std::unique_ptr<mlir::Pass> createStripAnnotationPass() { return std::make_unique<StripAnnotationPass>(); }
} // namespace conquer