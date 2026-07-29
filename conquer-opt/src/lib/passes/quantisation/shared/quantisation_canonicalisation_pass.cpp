#include "conquer/passes/quantisation/shared/quantisation_canonicalisation_pass.h"

#include "conquer/passes/quantisation/shared/patterns.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-quantisation-canonicalisation-pass"

void conquer::QuantisationCanonicalisationPass::getDependentDialects(mlir::DialectRegistry &registry) const {
    registry.insert<mlir::tosa::TosaDialect>();
}

void conquer::QuantisationCanonicalisationPass::runOnOperation() {
    const mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    populateTosaQuantisationCanonicalisationPatterns(patterns, quantPolicy);

    if (mlir::failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }
}