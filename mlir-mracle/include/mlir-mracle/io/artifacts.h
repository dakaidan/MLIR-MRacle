#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/io.h"

#include <string>

namespace mlir_mracle {

// saves the MLIR/LLVM artifacts produced during a run in a campaign for later possible inspection
void saveRunArtifacts(const RunInfo &run, const std::string &status,
                      const std::string &campaignDir,
                      bool unionFormat = false);

// writes the .ll artifact during execution mode
void saveExecutionArtifacts(const ExecutionRunResult &run,
                            const std::string &campaignDir);

// writes the campaign's result.json
void writeResultJson(const JsonValue &arr, const std::string &campaignDir);

// creates the campaign's fail/warn/ok status subfolders
void createCampaignStatusDirs(const std::string &campaignDir);

} // namespace mlir_mracle
