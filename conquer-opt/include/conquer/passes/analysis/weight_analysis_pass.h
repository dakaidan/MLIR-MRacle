#pragma once

#include "conquer/quantisation/stats.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <vector>

namespace conquer {

enum class WeightPartitionKind {
    None,
    Axis,
    DepthwiseCM
};

/// Holds the state for a single root constant calibration task.
struct ConstStatsTask {
    mlir::Operation *defOp;
    mlir::ElementsAttr tensorAttr;
    CalibrationStats globalStats;
    SensitivityStats globalSensitivity;
};

/// Holds the state for a single weight use calibration task.
struct WeightUseTask {
    mlir::Operation *rootConstOp;
    mlir::Operation *consumerOp;
    unsigned operandIndex;
    mlir::ElementsAttr tensorAttr;
    WeightPartitionKind partitionKind;
    int axis;
    CalibrationStats globalStats;
    std::vector<CalibrationStats> channelStats;
    SensitivityStats globalSensitivity;
};

/// An MLIR pass that annotates min, max, and scale values for quantisation calibration.
struct WeightAnalysisPass : public mlir::PassWrapper<WeightAnalysisPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WeightAnalysisPass)

    void getDependentDialects(mlir::DialectRegistry &registry) const override {}

    /// Returns the command-line argument that can be used to run this pass.
    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-weight-analysis"; }

    /// Returns a short description of the pass.
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Analyses weight tensors to determine quantisation parameters such as min, max, and scale.";
    }

    void runOnOperation() override;

  private:
    /// Determines the correct partition kind (e.g., per-tensor vs per-channel) for a specific consumer.
    static WeightPartitionKind getPartitionKind(mlir::Operation *consumerOp, unsigned operandIndex);

    /// Determines the correct axis to slice for per-channel quantization based on the consumer operation.
    static int getPerChannelAxis(mlir::Operation *consumerOp, unsigned operandIndex);

    /// Performs the heavy mathematical analysis off the main thread. Mutates ONLY the task object.
    static void analyseConstTask(ConstStatsTask &task);
    static void analyseUseTask(WeightUseTask &task);

    /// Takes the computed task stats and safely applies them to the MLIR IR on the main thread.
    static void applyConstStatsToIR(const ConstStatsTask &task, mlir::Builder &builder);
    static void applyUseStatsToIR(const WeightUseTask &task, mlir::Builder &builder);
};

inline std::unique_ptr<mlir::Pass> createWeightAnalysisPass() {
    return std::make_unique<WeightAnalysisPass>();
}
}