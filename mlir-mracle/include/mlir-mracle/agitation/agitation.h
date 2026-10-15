#pragma once

#include "llvm/IR/Module.h"

#include <cstdint>
#include <vector>

namespace mlir_mracle {

// Config for a single agitation run
struct AgitationConfig {
    int numThreads = 2; // number of threads to use for this run
    int runs = 0; // assigned by the experiment runner
};

// Generates a set of agitation configs for each binary
std::vector<AgitationConfig> generateAgitationConfigs(uint32_t seed,
                                                      int configCount);

// Shuffles the order of a function's non-entry basic blocks
void perturbBasicBlocks(llvm::Module &module, uint32_t seed);

} // namespace mlir_mracle
