#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

#include <string>
#include <vector>

namespace mlir_mracle {

// pipeline selectable via --mode; every requested mode runs in order
enum class PipelineMode {
    Emit,    // --mode=emit: transform and emit MLIR only
    Execute, // --mode=execution: execute each file as-is
    Legacy,  // --mode=legacy: legacy thread-group oracle pipeline
    Multi,   // --mode=multi: default agitation-sweep oracle pipeline
};

struct CliOptions {
    PipelineOptions pipeline;
    std::vector<PipelineMode> modes;
};

// Parses and validates every --... flag. Recognised flags are consumed and
// compacted out of argv (leftover arguments shift to the front, argc is
// reduced). On invalid input the historic error text is printed to stderr and
// the process exits with 1.
CliOptions parsePipelineOptions(int &argc, char **argv);

} // namespace mlir_mracle
