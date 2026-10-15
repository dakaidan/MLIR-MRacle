#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

#include <string>
#include <vector>

namespace mlir_mracle {

// pipeline chosen via --mode
enum class PipelineMode {
    Emit,      // --mode=emit: transform and emit MLIR
    Execute,   // --mode=execution: execute each file as-is
    Legacy,    // --mode=legacy: legacy thread-group oracle pipeline
    Compare,   // --mode=compare: default distributional oracle pipeline
};

struct CliOptions {
    PipelineOptions pipeline;
    std::vector<PipelineMode> modes;

    // flags only some pipelines honour, tracked so misuse is rejected instead
    // of silently ignored
    bool repsExplicit = false;
    bool rerunsExplicit = false;
    bool maxRunsExplicit = false;
    bool thresholdExplicit = false;
    bool campaignFlagsExplicit = false;   // any campaign-protocol flag was given
    bool executionsExplicit = false;      // --executions-per-campaign
    bool materialityExplicit = false;     // --tvd-threshold
    bool minChangeExplicit = false;       // --min-change
    bool maxCampaignsExplicit = false;    // --max-campaigns
    bool modeSelectsCalibrated = false;   // --mode=calibrated
    bool modeSelectsTvd = false;        // --mode=tvd was given
};

// Parses and validates every binary option flag and returns a CliOptions struct
CliOptions parsePipelineOptions(int &argc, char **argv);

} // namespace mlir_mracle
