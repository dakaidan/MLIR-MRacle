#include "mlir-mracle/agitation/agitation.h"
#include "mlir-mracle/backend/jit/jit.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <random>
#include <set>

namespace mlir_mracle {

namespace {

// Randomises the order of a function's non-entry basic blocks. Reordering
// basic blocks does not change the CFG, only the order in which they are
// emitted; the entry block must stay first for the verifier.
void shuffleBasicBlocks(llvm::Module &module, std::mt19937 &rng) {
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

} // namespace

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
        configs.push_back({{numThreads}, 0});
    }
    return configs;
}

std::vector<CompiledBinary> compileBinarySet(const llvm::Module &module,
                                             const CompileOptions &opts,
                                             std::string side,
                                             std::string *error) {
    if (error)
        error->clear();

    // every binary gets a CodeGen opt level from {0, 1, 2, 3}; with at least
    // four binaries all four levels are guaranteed to appear so the sweep
    // never misses a code generator. The order is shuffled from the seed so
    // source and transformed sides compile with the same sequence.
    std::vector<int> levels;
    levels.reserve(opts.binaryCount);
    for (int i = 0; i < opts.binaryCount; ++i)
        levels.push_back(i % 4);
    std::mt19937 optRng(opts.shuffleSeed);
    std::shuffle(levels.begin(), levels.end(), optRng);

    std::vector<CompiledBinary> out;
    for (int i = 0; i < opts.binaryCount; ++i) {
        int optLevel = levels[i];
        auto clone = llvm::CloneModule(module);
        llvm::BasicBlockSection bbSections = llvm::BasicBlockSection::None;
        if (opts.shuffleCode) {
            std::mt19937 rng(opts.shuffleSeed +
                             static_cast<uint32_t>(i) * 2654435761u);
            shuffleBasicBlocks(*clone, rng);
            bbSections = llvm::BasicBlockSection::All;
        }

        std::string err;
        auto fn = compileLLVMModuleToFunction(std::move(clone), &err,
                                              /*enableTsan=*/false, optLevel,
                                              bbSections);
        if (!fn) {
            if (error)
                *error = err;
            break;
        }
        out.push_back({side, i, optLevel, std::move(fn)});
    }
    return out;
}

} // namespace mlir_mracle
