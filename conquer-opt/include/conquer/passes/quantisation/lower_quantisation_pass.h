#pragma once

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace conquer {
    struct LowerQuantisationPass : mlir::PassWrapper<LowerQuantisationPass, mlir::OperationPass<mlir::ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQuantisationPass)

        LowerQuantisationPass() = default;

        void getDependentDialects(mlir::DialectRegistry &registry) const override;

        [[nodiscard]] llvm::StringRef getArgument() const override { return "conquer-lower-quantisation"; }
        [[nodiscard]] llvm::StringRef getDescription() const override {
            return "Lowers the quant dialect to TOSA";
        }

        void runOnOperation() override;
    };

    inline std::unique_ptr<mlir::Pass> createLowerQuantisationPass() {
        return std::make_unique<LowerQuantisationPass>();
    }
} // namespace conquer