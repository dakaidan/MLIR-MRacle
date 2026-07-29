#include "conquer/passes/quantisation/integer_quantisation_pass.h"
#include "conquer/core/logging.h"

#include "../../../../include/conquer/passes/quantisation/shared/patterns.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-integer-quantisation-pass"

void conquer::IntegerQuantisationPass::getDependentDialects(mlir::DialectRegistry &registry) const {
    registry.insert<mlir::tosa::TosaDialect>();
}

void conquer::IntegerQuantisationPass::runOnOperation() {
    const mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();

    L_INFO("Running Integer Quantisation Pass.");

    mlir::RewritePatternSet patterns(context);
    populateTosaIntegerQuantisationPatterns(patterns, quantPolicy);

    if (mlir::failed(applyPatternsGreedily(module, std::move(patterns)))) {
        L_DEBUG("  Integer quantisation pass failed.");
        signalPassFailure();
    } else {
        L_INFO("Integer Quantisation Pass succeeded.");
    }
}
