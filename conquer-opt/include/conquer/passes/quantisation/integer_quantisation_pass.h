#pragma once

#include "conquer/quantisation/policy.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <utility>

namespace conquer {
/// An MLIR pass that applies mixed-precision quantisation to operations
/// in a module based on a loaded quantisation policy.
struct IntegerQuantisationPass : mlir::PassWrapper<IntegerQuantisationPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerQuantisationPass)

    IntegerQuantisationPass() = default;
    explicit IntegerQuantisationPass(QuantisationPolicy policy) : quantPolicy(std::move(policy)) {}
    IntegerQuantisationPass(const IntegerQuantisationPass &other) = default;

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

inline std::unique_ptr<mlir::Pass> createIntegerQuantisationPass(const QuantisationPolicy &policy) {
    return std::make_unique<IntegerQuantisationPass>(policy);
}
} // namespace conquer