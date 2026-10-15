#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"

#include <string>
#include <vector>
#include <cmath>
#include <set>

namespace mlir_mracle {

// Which comparison the default pipeline runs
enum class ComparisonMethod {
    Agitation, // agitation sweep, replay rounds, TSan triage, per-outcome oracle
    Tvd,       // campaign-based distributional comparison, no triage
    Calibrated,// campaign protocol judged against a noise envelope it measures itself
};

// config for the runPipeline function
struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1;
    int runNumber = 0;          // first run index
    int numRuns = 1;            // pipeline repetitions, set by --iter
    int reps = 5000;            // emit mode: transformations picked; execution mode: runs per file
    std::string transform;      // comma-separated list of transforms to try, empty = any
    int maxApply = 1;           // limit of transforms to apply per run
    std::string model;          // memory model gating applicable transforms, empty = generic only
    std::string campaignDir;    // output folder; runs are added as they complete
    int retestReps = 5000;      // legacy pipeline replay budget
    int maxSourceReps = 100000; // legacy pipeline replay cap
    int thresholdPct = 5;       // share of runs a support violation must reach to fail a run

    // which comparison the default pipeline runs
    ComparisonMethod comparison = ComparisonMethod::Agitation;

    // measurement protocol of the TVD method
    int campaignsInitial = 10;  // campaigns collected before the first verdict,
                                // and the size of each later batch
    int campaignsMax = 50;      // cap while inconclusive: 50 x 2000 = 100,000
                                // executions per variant (30 x 1000 calibrated)
    int executionsPerCampaign = 2000;
    int teamSize = 4;           // OpenMP team size, identical for both variants
    double tvdThreshold = 0.05; // practical tolerance on the distributional discrepancy
    int bootstrapReplicates = 2000;
    double confidenceLevel = 0.95;

    // smallest distributional change the calibrated comparison reports, in
    // squared Hellinger distance
    double minChange = 0.005;

    // false-alarm budget of the calibrated comparison, shared across its looks
    double alphaTotal = 0.05;
};

// result of a single run of the pipeline, including the verdict and all outcome sets
struct PipelineResult {
    std::vector<RunInfo> runs;
    std::string campaignDir;
};

// core pipeline function
// applies transforms, runs the harness, and returns the verdict and all outcome sets
PipelineResult runPipeline(const PipelineOptions &opts);

// single run of the pipeline
// is also called by runPipeline for each repetition
RunInfo runSingle(const std::string &inputFile, int seed, int runIdx,
                  const PipelineOptions &opts);

// releases process-lifetime JIT state (compiled source binaries) while the
// runtime is still healthy; must be called before exit
void shutdownPipeline();

} // namespace mlir_mracle
