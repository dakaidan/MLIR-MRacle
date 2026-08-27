#include "mlir-mracle/agitation/agitation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <random>
#include <set>

namespace mlir_mracle {

void perturbBasicBlocks(llvm::Module &module, uint32_t seed) {
    std::mt19937 rng(seed);
    for (auto &func : module) {
        if (func.isDeclaration())
            continue;
        if (func.size() < 3)
            continue;
        llvm::SmallVector<llvm::BasicBlock *> tail;
        for (auto it = std::next(func.begin()); it != func.end(); ++it)
            tail.push_back(&*it);
        std::shuffle(tail.begin(), tail.end(), rng);
        llvm::BasicBlock *prev = &func.getEntryBlock();
        for (llvm::BasicBlock *bb : tail) {
            bb->moveAfter(prev);
            prev = bb;
        }
    }
}

std::vector<AgitationConfig> generateAgitationConfigs(uint32_t seed,
                                                      int configCount) {
    static constexpr int kThreads[] = {2, 3, 4, 6, 8};
    static constexpr int kCombos = 5;

    configCount = std::min(configCount, kCombos);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> threadDist(0, 4);

    std::vector<AgitationConfig> configs;
    configs.reserve(configCount);
    std::set<int> seen;
    while (static_cast<int>(configs.size()) < configCount) {
        int numThreads = kThreads[threadDist(rng)];
        if (!seen.insert(numThreads).second)
            continue;
        configs.push_back({numThreads, 0});
    }
    return configs;
}

} // namespace mlir_mracle
