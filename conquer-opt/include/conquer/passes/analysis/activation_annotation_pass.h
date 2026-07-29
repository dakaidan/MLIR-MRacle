#pragma once

#include "conquer/quantisation/stats.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <vector>

namespace conquer {
/// An MLIR pass that annotates min, max, and scale values for quantisation calibration.
struct ActivationAnnotationPass
    : public mlir::PassWrapper<ActivationAnnotationPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ActivationAnnotationPass)

    explicit ActivationAnnotationPass(std::vector<CalibrationStats> stats, std::vector<SensitivityStats> sensitivity_stats) : activationStats(std::move(stats)), sensitivityStats(std::move(sensitivity_stats)) {}

    ActivationAnnotationPass(const ActivationAnnotationPass &other) : activationStats(other.activationStats), sensitivityStats(other.sensitivityStats) {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {}

    /// Returns the command-line argument that can be used to run this pass.
    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-activation-annotation"; }
    /// Returns a short description of the pass.
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Annotates the original module with collected activation statistics from calibration.";
    }

    void runOnOperation() override;

  private:
    std::vector<CalibrationStats> activationStats;
    std::vector<SensitivityStats> sensitivityStats;
};

inline std::unique_ptr<mlir::Pass> createActivationAnnotationPass(std::vector<CalibrationStats> stats, std::vector<SensitivityStats> sensitivityStats) {
    return std::make_unique<ActivationAnnotationPass>(std::move(stats), std::move(sensitivityStats));
}
} // namespace conquer