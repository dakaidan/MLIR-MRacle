#include "mlir-mr/execution/execution.h"
#include "mlir-mr/agitation/agitation.h"
#include "mlir-mr/oracle/oracle.h"

#include <cstdlib>
#include <utility>

#include <omp.h>

namespace mlir_mr {

namespace {

omp_sched_t toOmpSched(ScheduleKind kind) {
    switch (kind) {
    case ScheduleKind::Static:
        return omp_sched_static;
    case ScheduleKind::Dynamic:
        return omp_sched_dynamic;
    case ScheduleKind::Guided:
        return omp_sched_guided;
    case ScheduleKind::Auto:
        return omp_sched_auto;
    }
    return omp_sched_static;
}

void applyProcessSettings() {
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_PROC_BIND", "close", 1);
    setenv("GOMP_CPU_AFFINITY", "0 1", 1);
}

void applyConfig(const AgitationConfig &cfg) {
    omp_set_dynamic(cfg.omp.dynamic ? 1 : 0);
    omp_set_num_threads(cfg.omp.numThreads);
    omp_set_schedule(toOmpSched(cfg.omp.schedule), cfg.omp.chunkSize);
}

BinaryExecutionResult runBinary(const CompiledBinary &binary,
                                const std::vector<AgitationConfig> &configs,
                                int singleThreadRuns, int binaryIndex) {
    BinaryExecutionResult result;
    result.binaryIndex = binaryIndex;
    result.side = binary.side;

    AgitationConfig single{{1, ScheduleKind::Static, 1, false},
                           singleThreadRuns};
    applyConfig(single);
    result.singleThread = collectOutcomeSet(binary.fn, singleThreadRuns, 1);

    for (size_t i = 0; i < configs.size(); ++i) {
        applyConfig(configs[i]);
        BinaryConfigResult cfgResult;
        cfgResult.configIndex = static_cast<int>(i);
        cfgResult.config = configs[i];
        cfgResult.outcomes = collectOutcomeSet(binary.fn, configs[i].runs,
                                               configs[i].omp.numThreads);
        result.perConfig.push_back(std::move(cfgResult));
        result.threadedTotal =
            mergeOutcomeSets(result.threadedTotal, cfgResult.outcomes);
    }
    return result;
}

} // namespace

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
    return result;
}

} // namespace mlir_mr
