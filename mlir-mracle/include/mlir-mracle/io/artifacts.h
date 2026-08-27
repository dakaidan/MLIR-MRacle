#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/io.h"

#include <string>

namespace mlir_mracle {

// writes the artifacts of a single run under
// <campaignDir>/<status>/run<N>_seed<S>/ so a long campaign publishes each
// run's output as it completes instead of buffering everything until the end.
// unionFormat selects the default run_info.json layout (union outcome sets
// plus per-binary breakdown); the alternative writes the thread-group layout.
void saveRunArtifacts(const RunInfo &run, const std::string &status,
                      const std::string &campaignDir,
                      bool unionFormat = false);

// writes the .ll artifact of one execution-mode run under
// <campaignDir>/run<N>_seed<S>/
void saveExecutionArtifacts(const ExecutionRunResult &run,
                            const std::string &campaignDir);

// writes the campaign's result.json
void writeResultJson(const JsonValue &arr, const std::string &campaignDir);

// creates the campaign's fail/warn/ok status subfolders
void createCampaignStatusDirs(const std::string &campaignDir);

} // namespace mlir_mracle
