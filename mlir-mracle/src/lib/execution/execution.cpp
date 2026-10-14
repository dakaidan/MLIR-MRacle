#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/backend/jit/jit.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

#include <omp.h>
#include <vector>

namespace mlir_mracle {

void applyProcessSettings(int teamSize) {
    setenv("OMP_DYNAMIC", "false", 1);

    const std::string threads = std::to_string(teamSize > 0 ? teamSize : 1);
    setenv("OMP_NUM_THREADS", threads.c_str(), 1);
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

namespace {

// compiles and runs one campaign per side under the fixed protocol
bool runPair(ExecutionResult &exec, const llvm::Module &sourceModule,
             const llvm::Module &transformedModule, int campaignIndex,
             uint32_t campaignSeed, const CampaignOptions &opts,
             std::string &error) {
    error.clear();

    if (opts.executionsPerCampaign <= 0 || opts.teamSize <= 0) {
        error = "campaign executions and team size must be positive";
        return false;
    }

    applyProcessSettings(opts.teamSize);

    // sides alternate so load and ordering effects are not tied to one variant
    const bool sourceFirst = (campaignSeed % 2u) == 0u;
    const int sideOrder[2] = {sourceFirst ? 0 : 1, sourceFirst ? 1 : 0};

    std::vector<CompiledBinary> compiled(2);
    for (int step = 0; step < 2; ++step) {
        const int side = sideOrder[step];
        const llvm::Module &module = side == 0 ? sourceModule : transformedModule;
        const std::string sideName = side == 0 ? "source" : "transformed";

        std::string compileError;
        auto fn = compileLLVMModuleToFunction(
            llvm::CloneModule(module), &compileError, /*enableTsan=*/false,
            opts.jitOptLevel, llvm::BasicBlockSection::None);
        if (!fn) {
            error = "JIT compile error (" + sideName + "): " + compileError;
            return false;
        }

        compiled[side].identity = {sideName, campaignIndex, opts.jitOptLevel,
                                   campaignSeed};
        compiled[side].fn = std::move(fn);

        // the first campaign also probes both sides at one thread, before any
        // threaded execution, so a determinism mismatch is caught early
        if (campaignIndex == 0 && opts.determinismProbeRuns > 0) {
            ObservedOutcomeSet probe = collectOutcomeSet(
                compiled[side].fn, opts.determinismProbeRuns, /*numThreads=*/1);
            if (side == 0)
                exec.sourceSingleThreadTotal = mergeOutcomeSets(
                    exec.sourceSingleThreadTotal, probe);
            else
                exec.transformedSingleThreadTotal = mergeOutcomeSets(
                    exec.transformedSingleThreadTotal, probe);
        }

        compiled[side].threadedTotal =
            collectOutcomeSet(compiled[side].fn, opts.executionsPerCampaign,
                              opts.teamSize);
    }

    // campaign boundaries are kept: each pair is appended as two records, one
    // per side, and the aggregate sets are only used for reporting
    for (int side = 0; side < 2; ++side) {
        const ObservedOutcomeSet &set = compiled[side].threadedTotal;
        if (side == 0) {
            exec.sourceTotal = mergeOutcomeSets(exec.sourceTotal, set);
            exec.sourceBinaries.push_back(std::move(compiled[side]));
        } else {
            exec.transformedTotal = mergeOutcomeSets(exec.transformedTotal, set);
            exec.transformedBinaries.push_back(std::move(compiled[side]));
        }
    }

    exec.total = mergeOutcomeSets(exec.sourceTotal, exec.transformedTotal);
    return true;
}

} // namespace

bool runCampaignPair(ExecutionResult &exec, const llvm::Module &sourceModule,
                     const llvm::Module &transformedModule, int campaignIndex,
                     uint32_t campaignSeed, const CampaignOptions &opts,
                     std::string &error) {
    return runPair(exec, sourceModule, transformedModule, campaignIndex,
                   campaignSeed, opts, error);
}


} // namespace mlir_mracle
