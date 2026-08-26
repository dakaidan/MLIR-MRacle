#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/agitation/agitation.h"
#include "mlir-mracle/backend/jit/jit.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdlib>
#include <map>
#include <utility>

#include <omp.h>

namespace mlir_mracle {

namespace {

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

void applyProcessSettings() {
    // OMP_DYNAMIC and OMP_NUM_THREADS are read only at OpenMP runtime
    // initialisation, so they are pinned once process-wide; per-config
    // variation uses omp_set_num_threads only.
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
}

void distributeRuns(std::vector<AgitationConfig> &configs, int totalRuns) {
    int base = totalRuns / static_cast<int>(configs.size());
    int rem = totalRuns % static_cast<int>(configs.size());
    for (size_t i = 0; i < configs.size(); ++i)
        configs[i].runs = base + (static_cast<int>(i) < rem ? 1 : 0);
}

std::vector<const CompiledBinary *>
collectAllBinaries(const ExecutionResult &exec) {
    std::vector<const CompiledBinary *> binaries;
    binaries.reserve(exec.sourceBinaries.size() +
                     exec.transformedBinaries.size());
    for (const auto &b : exec.sourceBinaries)
        binaries.push_back(&b);
    for (const auto &b : exec.transformedBinaries)
        binaries.push_back(&b);
    return binaries;
}

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
    distributeRuns(configs, opts.runsPerBinary);

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

    distributeRuns(runConfigs, extraRuns);

    std::vector<const CompiledBinary *> binaries = collectAllBinaries(exec);

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

    // one TSan-instrumented binary per side; the agitation sweep already
    // sampled the codegen axis plain, so the triage only adds the scheduling
    // perturbation axis
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
            compileError = err;
            break;
        }
        tsanBinaries.push_back({entry.second, 0, 2, std::move(fn)});
    }
    if (!compileError.empty()) {
        if (error)
            *error = compileError;
        return false;
    }

    applyProcessSettings();
    std::vector<BinaryExecutionResult> triageResults;
    triageResults.reserve(2);
    int binaryIndex = static_cast<int>(exec.sourceBinaries.size() +
                                       exec.transformedBinaries.size());
    for (const auto &binary : tsanBinaries) {
        // single-thread probes stay with the plain binaries: the determinism
        // check must not see the TSan perturbation
        triageResults.push_back(runBinary(binary, triageConfigs,
                                          /*singleThreadRuns=*/0,
                                          binaryIndex++));
    }

    // keep binaryResults grouped source-then-transformed so the per-binary
    // outcome breakdown maps back to the binary lists by index; the source
    // triage result is spliced in before the transformed section
    size_t sourceEnd = exec.sourceBinaries.size();
    for (size_t i = 0; i < triageResults.size(); ++i) {
        BinaryExecutionResult &br = triageResults[i];
        if (br.side == "source") {
            exec.sourceBinaries.push_back(tsanBinaries[i]);
            exec.sourceTotal =
                mergeOutcomeSets(exec.sourceTotal, br.threadedTotal);
            exec.binaryResults.insert(exec.binaryResults.begin() + sourceEnd,
                                      std::move(br));
            ++sourceEnd;
        } else {
            exec.transformedBinaries.push_back(tsanBinaries[i]);
            exec.transformedTotal =
                mergeOutcomeSets(exec.transformedTotal, br.threadedTotal);
            exec.binaryResults.push_back(std::move(br));
        }
    }
    exec.total = mergeOutcomeSets(exec.sourceTotal, exec.transformedTotal);
    return true;
}

} // namespace mlir_mracle
