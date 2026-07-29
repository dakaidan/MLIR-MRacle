#include "conquer/passes/quantisation/float_quantisation_pass.h"
#include "conquer/core/logging.h"

#include "../../../../include/conquer/passes/quantisation/shared/patterns.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-float-quantisation-pass"

void conquer::FloatQuantisationPass::getDependentDialects(mlir::DialectRegistry &registry) const {
    registry.insert<mlir::tosa::TosaDialect>();
}

void conquer::FloatQuantisationPass::runOnOperation() {
    const mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();

    L_INFO("Running Float Quantisation Pass.");

    mlir::RewritePatternSet patterns(context);
    populateTosaFloatQuantisationPatterns(patterns, quantPolicy);

    if (mlir::failed(applyPatternsGreedily(module, std::move(patterns)))) {
        L_DEBUG("  Float quantisation pass failed.");
        signalPassFailure();
    } else {
        L_INFO("Float Quantisation Pass succeeded.");
    }
}
