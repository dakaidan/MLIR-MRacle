#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/agitation/agitation.h"

#include <cstdlib>
#include <map>
#include <utility>

#include <omp.h>

namespace mlir_mracle {

namespace {

void applyProcessSettings() {
    // OMP_DYNAMIC is read only at OpenMP runtime initialisation, so it is
    // pinned once process-wide; per-config variation uses
    // omp_set_num_threads only.
    setenv("OMP_DYNAMIC", "false", 1);
}

void applyConfig(const AgitationConfig &cfg) {
    omp_set_num_threads(cfg.omp.numThreads);
}

BinaryExecutionResult runBinary(const CompiledBinary &binary,
                                const std::vector<AgitationConfig> &configs,
                                int singleThreadRuns, int binaryIndex) {
    BinaryExecutionResult result;
    result.binaryIndex = binaryIndex;
    result.side = binary.side;

    AgitationConfig single{{1}, singleThreadRuns};
    applyConfig(single);
    result.singleThread = collectOutcomeSet(binary.fn, singleThreadRuns, 1);

    for (size_t i = 0; i < configs.size(); ++i) {
        applyConfig(configs[i]);
        BinaryConfigResult cfgResult;
        cfgResult.configIndex = static_cast<int>(i);
        cfgResult.config = configs[i];
        cfgResult.outcomes = collectOutcomeSet(binary.fn, configs[i].runs,
                                               configs[i].omp.numThreads);
        result.threadedTotal =
            mergeOutcomeSets(result.threadedTotal, cfgResult.outcomes);
        result.perConfig.push_back(std::move(cfgResult));
    }
    return result;
}

} // namespace

ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads) {
    // a single caller thread owns the whole sweep; numThreads only sizes the
    // OpenMP team the program runs with, so the harness itself cannot race
    // on the JIT'd module's globals or the OpenMP runtime
    omp_set_num_threads(numThreads);
    ObservedOutcomeSet set;
    std::map<JointOutcome, int64_t> freq;
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
    auto configs = generateAgitationConfigs(opts.seed, opts.configCount);
    if (configs.empty()) {
        result.error = "no agitation configs generated";
        return result;
    }
    int base = opts.runsPerBinary / static_cast<int>(configs.size());
    int rem = opts.runsPerBinary % static_cast<int>(configs.size());
    for (size_t i = 0; i < configs.size(); ++i)
        configs[i].runs = base + (static_cast<int>(i) < rem ? 1 : 0);

    applyProcessSettings();

    result.sourceBinaries = compileBinarySet(sourceModule, opts.compile,
                                             "source", &result.error);
    result.transformedBinaries = compileBinarySet(transformedModule,
                                                  opts.compile, "transformed",
                                                  &result.error);
    if (result.sourceBinaries.empty() && result.transformedBinaries.empty())
        return result;

    int binaryIndex = 0;
    for (const auto &binary : result.sourceBinaries) {
        auto br = runBinary(binary, configs, opts.singleThreadRuns,
                            binaryIndex++);
        result.sourceTotal =
            mergeOutcomeSets(result.sourceTotal, br.threadedTotal);
        result.sourceSingleThreadTotal =
            mergeOutcomeSets(result.sourceSingleThreadTotal, br.singleThread);
        result.binaryResults.push_back(std::move(br));
    }
    for (const auto &binary : result.transformedBinaries) {
        auto br = runBinary(binary, configs, opts.singleThreadRuns,
                            binaryIndex++);
        result.transformedTotal =
            mergeOutcomeSets(result.transformedTotal, br.threadedTotal);
        result.transformedSingleThreadTotal =
            mergeOutcomeSets(result.transformedSingleThreadTotal,
                             br.singleThread);
        result.binaryResults.push_back(std::move(br));
    }
    result.total = mergeOutcomeSets(result.sourceTotal, result.transformedTotal);
    result.configs = std::move(configs);
    return result;
}

void rerunAllBinaries(ExecutionResult &exec, int extraRuns,
                      uint32_t roundSeed, int configCount) {
    if (extraRuns <= 0)
        return;

    // a replay round re-agitates: draw a fresh team-size mix from the round
    // seed instead of re-using the initial batch's configs
    std::vector<AgitationConfig> runConfigs =
        generateAgitationConfigs(roundSeed, configCount);
    if (runConfigs.empty())
        return;

    int base = extraRuns / static_cast<int>(runConfigs.size());
    int rem = extraRuns % static_cast<int>(runConfigs.size());
    for (size_t i = 0; i < runConfigs.size(); ++i)
        runConfigs[i].runs = base + (static_cast<int>(i) < rem ? 1 : 0);

    std::vector<const CompiledBinary *> binaries;
    binaries.reserve(exec.sourceBinaries.size() +
                     exec.transformedBinaries.size());
    for (const auto &b : exec.sourceBinaries)
        binaries.push_back(&b);
    for (const auto &b : exec.transformedBinaries)
        binaries.push_back(&b);

    // the per-config breakdown describes the initial batch only; replay
    // rounds merge straight into the per-binary threaded totals
    for (size_t b = 0; b < binaries.size() && b < exec.binaryResults.size();
         ++b) {
        auto &br = exec.binaryResults[b];
        for (const auto &cfg : runConfigs) {
            applyConfig(cfg);
            ObservedOutcomeSet extraSet = collectOutcomeSet(
                binaries[b]->fn, cfg.runs, cfg.omp.numThreads);
            br.threadedTotal =
                mergeOutcomeSets(br.threadedTotal, extraSet);
        }
    }

    exec.sourceTotal = ObservedOutcomeSet{};
    exec.transformedTotal = ObservedOutcomeSet{};
    for (const auto &br : exec.binaryResults) {
        if (br.side == "source")
            exec.sourceTotal =
                mergeOutcomeSets(exec.sourceTotal, br.threadedTotal);
        else
            exec.transformedTotal =
                mergeOutcomeSets(exec.transformedTotal, br.threadedTotal);
    }
    exec.total = mergeOutcomeSets(exec.sourceTotal, exec.transformedTotal);
}

} // namespace mlir_mracle
