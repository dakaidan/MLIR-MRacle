#pragma once

#include "conquer/quantisation/policy.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <utility>

namespace conquer {
/// An MLIR pass that applies mixed-precision quantisation to operations
/// in a module based on a loaded quantisation policy.
struct FloatQuantisationPass : mlir::PassWrapper<FloatQuantisationPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FloatQuantisationPass)

    FloatQuantisationPass() = default;
    explicit FloatQuantisationPass(QuantisationPolicy policy) : quantPolicy(std::move(policy)) {}
    FloatQuantisationPass(const FloatQuantisationPass &other) = default;

    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void setQuantisationPolicy(const QuantisationPolicy &policy) { quantPolicy = policy; }

    [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-quantisation"; }
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Applies mixed-precision quantisation to the model based on the loaded policy";
    }

    void runOnOperation() override;

  private:
    QuantisationPolicy quantPolicy;
};

inline std::unique_ptr<mlir::Pass> createFloatQuantisationPass(const QuantisationPolicy &policy) {
    return std::make_unique<FloatQuantisationPass>(policy);
}
} // namespace conquer