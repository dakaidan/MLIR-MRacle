#include "conquer/core/logging.h"
#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-quantisation-patterns"

#include "conquer/passes/quantisation/shared/patterns.h"

#include "conquer/passes/quantisation/float_patterns/absorb.h"
#include "conquer/passes/quantisation/float_patterns/compute.h"
#include "conquer/passes/quantisation/float_patterns/elementwise.h"
#include "conquer/passes/quantisation/float_patterns/fft.h"
#include "conquer/passes/quantisation/float_patterns/reduction.h"
#include "conquer/passes/quantisation/float_patterns/index.h"

#include "conquer/passes/quantisation/shared/canonicalisation_patterns.h"

#include "conquer/passes/quantisation/integer_patterns/compute.h"
#include "conquer/passes/quantisation/integer_patterns/elementwise.h"
#include "conquer/passes/quantisation/integer_patterns/table.h"
#include "conquer/passes/quantisation/integer_patterns/layout.h"
#include "conquer/passes/quantisation/integer_patterns/pooling.h"
#include "conquer/passes/quantisation/integer_patterns/fft.h"


void conquer::populateTosaFloatQuantisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    L_DEBUG("Populating TOSA float quantisation patterns.");
    float_quant::populateAbsorbPatterns(patterns);
    float_quant::populateComputePatterns(patterns, policy);
    float_quant::populateElementWisePatterns(patterns, policy);
    float_quant::populateFFTPatterns(patterns, policy);
    float_quant::populateReductionPatterns(patterns, policy);
    float_quant::populateIndexPatterns(patterns, policy);
}

void conquer::populateTosaIntegerQuantisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    L_DEBUG("Populating TOSA integer quantisation patterns.");
    integer_quant::populateComputePatterns(patterns, policy);
    integer_quant::populateElementWisePatterns(patterns, policy);
    integer_quant::populateTablePatterns(patterns);
    integer_quant::populateLayoutPatterns(patterns);
    integer_quant::populatePoolingPatterns(patterns, policy);
    integer_quant::populateFFTPatterns(patterns, policy);
}

void conquer::populateTosaQuantisationCanonicalisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy) {
    L_DEBUG("Populating TOSA quantisation canonicalisation patterns.");
    populateCanonicalisePatterns(patterns, policy);
}
