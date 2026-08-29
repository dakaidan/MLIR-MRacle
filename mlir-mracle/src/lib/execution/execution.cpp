#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/agitation/agitation.h"
#include "mlir-mracle/backend/jit/jit.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <random>
#include <utility>

#include <omp.h>
#include <vector>

namespace mlir_mracle {

namespace {

void applyConfig(const AgitationConfig &cfg) {
    omp_set_num_threads(cfg.numThreads);
}

// Runs a single compiled binary under the agitation configs and collects, merges the outcome sets
void runBinary(CompiledBinary &binary,
               const std::vector<AgitationConfig> &configs,
               int singleThreadRuns) {
    AgitationConfig single{1, singleThreadRuns};
    applyConfig(single);
    binary.singleThread = collectOutcomeSet(binary.fn, singleThreadRuns, 1);

    for (const auto &cfg : configs) {
        applyConfig(cfg);
        ObservedOutcomeSet set = collectOutcomeSet(binary.fn, cfg.runs, cfg.numThreads);
        binary.threadedTotal = mergeOutcomeSets(binary.threadedTotal, set);
    }
}

// Builds the 5 in-memory binary set for one side
std::vector<CompiledBinary> compileBinarySet(const llvm::Module &module,
                                             uint32_t seed, std::string side,
                                             std::string *error) {
    if (error)
        error->clear();

    constexpr int kBinaryCount = 5;
    std::vector<int> levels;
    levels.reserve(kBinaryCount);

    for (int i = 0; i < kBinaryCount; ++i)
        levels.push_back(i % 4);

    std::mt19937 optRng(seed);
    std::shuffle(levels.begin(), levels.end(), optRng);

    std::vector<CompiledBinary> out;

    // loop to compile 5 binaries with different jitOptLevel and basic-block layout
    for (int i = 0; i < kBinaryCount; ++i) {
        int optLevel = levels[i];
        auto clone = llvm::CloneModule(module);
        perturbBasicBlocks(*clone, seed + static_cast<uint32_t>(i) * 2654435761u);

        std::string err;
        auto fn = compileLLVMModuleToFunction(std::move(clone), &err,
                                              /*enableTsan=*/false, optLevel,
                                              llvm::BasicBlockSection::All);

        if (!fn) {
            if (error)
                *error = "JIT compile error (" + side + "): " + err;
            break;
        }

        CompiledBinary binary;
        binary.identity = {side, i, optLevel};
        binary.fn = std::move(fn);
        out.push_back(std::move(binary));
    }
    return out;
}

} // namespace

void applyProcessSettings() {
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
}

void distributeRuns(std::vector<AgitationConfig> &configs, int totalRuns) {
    int base = totalRuns / static_cast<int>(configs.size());
    int rem = totalRuns % static_cast<int>(configs.size());

    for (size_t i = 0; i < configs.size(); ++i)
        configs[i].runs = base + (static_cast<int>(i) < rem ? 1 : 0);
}

ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads) {
    
    // numThreads only sizes the OpenMP team the program runs with
    // so the harness itself cannot race on the JIT'd module's globals or the OpenMP runtime
    omp_set_num_threads(numThreads);
    
    ObservedOutcomeSet set;
    std::map<JointOutcome, int64_t> freq;

    // run the binary numRuns times and collect the joint outcomes
    for (int i = 0; i < numRuns; ++i) {
        auto res = fn();

        if (set.arityConsistent) {
            if (set.arity == 0)
                set.arity = res.size();
            else if (res.size() != set.arity)
                set.arityConsistent = false;
        }

        ++freq[std::move(res)];
    }

    // fill the ObservedOutcomeSet with the sorted unique joint outcomes and their counts
    set.outcomes.reserve(freq.size());
    set.counts.reserve(freq.size());

    for (const auto &[outcome, count] : freq) {
        set.outcomes.push_back(outcome);
        set.counts.push_back(count);
    }

    set.totalRuns = numRuns;
    return set;
}

ObservedOutcomeSet mergeOutcomeSets(const ObservedOutcomeSet &a,
                                    const ObservedOutcomeSet &b) {
    ObservedOutcomeSet out;
    out.totalRuns = a.totalRuns + b.totalRuns;
    out.outcomes.reserve(a.outcomes.size() + b.outcomes.size());
    out.counts.reserve(a.counts.size() + b.counts.size());

    size_t i = 0, j = 0;
    while (i < a.outcomes.size() || j < b.outcomes.size()) {
        if (j >= b.outcomes.size() ||
            (i < a.outcomes.size() && a.outcomes[i] < b.outcomes[j])) {
            out.outcomes.push_back(a.outcomes[i]);
            out.counts.push_back(i < a.counts.size() ? a.counts[i] : 0);
            ++i;
        } else if (i >= a.outcomes.size() ||
                   b.outcomes[j] < a.outcomes[i]) {
            out.outcomes.push_back(b.outcomes[j]);
            out.counts.push_back(j < b.counts.size() ? b.counts[j] : 0);
            ++j;
        } else {
            out.outcomes.push_back(a.outcomes[i]);
            out.counts.push_back((i < a.counts.size() ? a.counts[i] : 0) +
                                 (j < b.counts.size() ? b.counts[j] : 0));
            ++i;
            ++j;
        }
    }

    out.arityConsistent = true;
    for (const auto &jo : out.outcomes) {
        if (out.arity == 0)
            out.arity = jo.size();
        else if (jo.size() != out.arity)
            out.arityConsistent = false;
    }
    return out;
}

ExecutionResult runExecutionHarness(const llvm::Module &sourceModule,
                                    const llvm::Module &transformedModule,
                                    const ExecutionOptions &opts) {
    ExecutionResult result;
    if (opts.runsPerBinary <= 0 || opts.singleThreadRuns <= 0) {
        result.error = "runs must be positive";
        return result;
    }

    // generate agitation configs and distribute the runs across them
    auto configs = generateAgitationConfigs(opts.seed, opts.configCount);
    if (configs.empty()) {
        result.error = "no agitation configs generated";
        return result;
    }
    distributeRuns(configs, opts.runsPerBinary);
    applyProcessSettings();

    // compile the two cloned binary sets; a compile failure on either side
    // is fatal because the comparison needs both sides
    std::string sourceError, transformedError;
    result.sourceBinaries = compileBinarySet(sourceModule, opts.seed, "source",
                                             &sourceError);
    result.transformedBinaries = compileBinarySet(transformedModule, opts.seed,
                                                  "transformed",
                                                  &transformedError);
    if (!sourceError.empty() || !transformedError.empty()) {
        result.error = !sourceError.empty() ? sourceError : transformedError;
        return result;
    }

    // run every binary under the agitation configs and merge the outcome sets
    for (auto &binary : result.sourceBinaries) {
        runBinary(binary, configs, opts.singleThreadRuns);
        result.sourceTotal =
            mergeOutcomeSets(result.sourceTotal, binary.threadedTotal);
        result.sourceSingleThreadTotal =
            mergeOutcomeSets(result.sourceSingleThreadTotal,
                             binary.singleThread);
    }
    
    for (auto &binary : result.transformedBinaries) {
        runBinary(binary, configs, opts.singleThreadRuns);
        result.transformedTotal =
            mergeOutcomeSets(result.transformedTotal, binary.threadedTotal);
        result.transformedSingleThreadTotal =
            mergeOutcomeSets(result.transformedSingleThreadTotal,
                             binary.singleThread);
    }
    result.total = mergeOutcomeSets(result.sourceTotal, result.transformedTotal);
    return result;
}

void rerunAllBinaries(ExecutionResult &exec, int extraRuns,
                      uint32_t roundSeed, int configCount) {
    if (extraRuns <= 0)
        return;

    // a replay round re-agitates
    std::vector<AgitationConfig> runConfigs =
        generateAgitationConfigs(roundSeed, configCount);

    if (runConfigs.empty())
        return;

    distributeRuns(runConfigs, extraRuns);

    auto rerunSide = [&](std::vector<CompiledBinary> &binaries) {
        for (auto &binary : binaries)
            for (const auto &cfg : runConfigs) {
                applyConfig(cfg);
                ObservedOutcomeSet extraSet = collectOutcomeSet(
                    binary.fn, cfg.runs, cfg.numThreads);
                binary.threadedTotal =
                    mergeOutcomeSets(binary.threadedTotal, extraSet);
            }
    };

    // both sides are re-run and re-agitated
    rerunSide(exec.sourceBinaries);
    rerunSide(exec.transformedBinaries);

    exec.sourceTotal = ObservedOutcomeSet{};
    exec.transformedTotal = ObservedOutcomeSet{};

    // merge the per-binary totals into the aggregate outcome sets again
    for (const auto &binary : exec.sourceBinaries)
        exec.sourceTotal =
            mergeOutcomeSets(exec.sourceTotal, binary.threadedTotal);
    for (const auto &binary : exec.transformedBinaries)
        exec.transformedTotal =
            mergeOutcomeSets(exec.transformedTotal, binary.threadedTotal);
    exec.total = mergeOutcomeSets(exec.sourceTotal, exec.transformedTotal);
}

bool runTsanTriage(const llvm::Module &sourceModule,
                   const llvm::Module &transformedModule,
                   ExecutionResult &exec, int extraRuns, uint32_t triageSeed,
                   int configCount, std::string *error) {
    if (error)
        error->clear();

    if (extraRuns <= 0)
        return true;

    std::vector<AgitationConfig> triageConfigs =
        generateAgitationConfigs(triageSeed, configCount);

    if (triageConfigs.empty())
        return true;

    distributeRuns(triageConfigs, extraRuns);

    // one TSan-instrumented binary per side
    std::string compileError;
    std::vector<CompiledBinary> tsanBinaries;

    tsanBinaries.reserve(2);

    for (const auto &entry :
         {std::pair<const llvm::Module *, std::string>{
              &sourceModule, "source"},
          {&transformedModule, "transformed"}}) {
        std::string err;
        auto fn = compileLLVMModuleToFunction(llvm::CloneModule(*entry.first),
                                              &err, /*enableTsan=*/true,
                                              /*jitOptLevel=*/2);
        if (!fn) {
            compileError =
                "JIT compile error (" + entry.second + "): " + err;
            break;
        }

        CompiledBinary binary;
        binary.identity = {entry.second, 0, 2};
        binary.fn = std::move(fn);
        tsanBinaries.push_back(std::move(binary));
    }

    if (!compileError.empty()) {
        if (error)
            *error = compileError;
        return false;
    }

    applyProcessSettings();

    // single-thread probes stay with the plain binaries
    for (auto &binary : tsanBinaries)
        runBinary(binary, triageConfigs, /*singleThreadRuns=*/0);

    // merge the TSan-instrumented binaries' totals into the aggregate outcome sets
    for (auto &binary : tsanBinaries) {
        if (binary.identity.side == "source") {
            exec.sourceTotal =
                mergeOutcomeSets(exec.sourceTotal, binary.threadedTotal);
            exec.sourceBinaries.push_back(std::move(binary));
        } else {
            exec.transformedTotal =
                mergeOutcomeSets(exec.transformedTotal, binary.threadedTotal);
            exec.transformedBinaries.push_back(std::move(binary));
        }
    }
    
    exec.total = mergeOutcomeSets(exec.sourceTotal, exec.transformedTotal);
    return true;
}

} // namespace mlir_mracle
