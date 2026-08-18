#include "mlir-mr/agitation/agitation.h"
#include "mlir-mr/backend/jit/jit.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <random>
#include <set>
#include <tuple>

namespace mlir_mr {

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
    static constexpr ScheduleKind kSchedules[] = {
        ScheduleKind::Static, ScheduleKind::Dynamic, ScheduleKind::Guided,
        ScheduleKind::Auto};
    static constexpr int kChunks[] = {1, 2, 4, 8, 16};
    static constexpr int kCombos = 5 * 4 * 5 * 2; // threads * scheds * chunks * dynamic

    configCount = std::min(configCount, kCombos);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> threadDist(0, 4);
    std::uniform_int_distribution<int> schedDist(0, 3);
    std::uniform_int_distribution<int> chunkDist(0, 4);
    std::uniform_int_distribution<int> dynDist(0, 1);

    std::vector<AgitationConfig> configs;
    configs.reserve(configCount);
    std::set<std::tuple<int, int, int, int>> seen;
    while (static_cast<int>(configs.size()) < configCount) {
        OpenMPSettings omp{kThreads[threadDist(rng)],
                           kSchedules[schedDist(rng)], kChunks[chunkDist(rng)],
                           dynDist(rng) == 1};
        auto key = std::make_tuple(omp.numThreads,
                                   static_cast<int>(omp.schedule),
                                   omp.chunkSize, omp.dynamic ? 1 : 0);
        if (!seen.insert(key).second)
            continue;
        configs.push_back({omp, 0});
    }
    return configs;
}

std::vector<CompiledBinary> compileBinarySet(const llvm::Module &module,
                                             const CompileOptions &opts,
                                             std::string side,
                                             std::string *error) {
    if (error)
        error->clear();

    std::vector<int> levels = opts.jitOptLevels;
    if (levels.empty())
        levels = {0, 1, 2, 3, 3};

    std::vector<CompiledBinary> out;
    for (size_t i = 0; i < levels.size(); ++i) {
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
                                              opts.enableTsan, levels[i],
                                              bbSections);
        if (!fn) {
            if (error)
                *error = err;
            break;
        }
        out.push_back({side, static_cast<int>(i), levels[i], std::move(fn)});
    }
    return out;
}

} // namespace mlir_mr
