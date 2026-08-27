#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"

#include <string>
#include <vector>
#include <cmath>

namespace mlir_mracle {

// config for the runPipeline function
struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1; // -1 = random base seed drawn per process, mixed with the run index; >=0 = fixed
    int runNumber = 0;        // first run index
    int numRuns = 1;          // pipeline repetitions, set by --iter
    int reps = 5000;           // --reps: executions per program per thread count
    std::string transform;    // comma-separated list of transforms to try, empty = any
    int maxApply = 1;         // limit of transforms to apply per run
    std::string model;        // memory model gating applicable transforms, empty = generic only
    std::string campaignDir;  // output folder; runs are added as they complete
    int retestReps = 5000;    // extra source runs when transformed finds a new outcome
    int maxSourceReps = 100000; // hard cap for total source runs across all binaries
    int thresholdPct = 5; // fail/warn classifier; deviations are only flagged when Poisson-significant
};

struct PipelineResult {
    std::vector<RunInfo> runs;
    std::string campaignDir;
};

// default mode: compile both modules as an in-memory binary set (no
// bitcode/cache), run the agitation sweep, judge the aggregate outcome sets
// with oracleCompare, replay rare states up to maxSourceReps, then report
// the final post-replay verdict.
PipelineResult runPipeline(const PipelineOptions &opts);

// single run of the default pipeline: applies the requested transforms, adds
// symmetric jitter delay chains to both modules, lowers and translates both
// modules directly (no persistent cache, no source memo), runs the agitation
// sweep through the harness, then replays rare states in rounds of --reruns
// until they resolve or the total source runs across all binaries reach the
// --max-runs cap, with a final TSan-instrumented triage when the cap is
// reached unresolved. The verdict is the final post-replay comparison,
// judged on merged data.
RunInfo runSingle(const std::string &inputFile, int seed, int runIdx,
                  const PipelineOptions &opts);

// releases process-lifetime JIT state (compiled source binaries) while the
// runtime is still healthy; must be called before exit
void shutdownPipeline();

} // namespace mlir_mracle
