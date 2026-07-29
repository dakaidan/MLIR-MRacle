#pragma once

#include "conquer/quantisation/policy.h"

#include <mlir/IR/PatternMatch.h>

namespace conquer {
void populateTosaFloatQuantisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy);
void populateTosaIntegerQuantisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy);
void populateTosaQuantisationCanonicalisationPatterns(mlir::RewritePatternSet &patterns, const QuantisationPolicy &policy);
}