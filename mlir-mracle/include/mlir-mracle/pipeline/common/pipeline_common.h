#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir_mracle {

// the 1-thread group is a determinism check, so a small budget suffices
inline constexpr int kDeterminismReps = 32;

// team-size sweep per run; the 2-thread primary test runs first so it is
// always the first to fail or warn. 1 is the single-thread determinism
// probe, 4/8 the wider concurrency sweep.
inline constexpr int kThreadCounts[] = {2, 1, 4, 8};

// the legacy --tsan pipeline and execution mode keep the JIT's default
// CodeGen opt level; the default agitation sweep randomises opt levels per
// compiled binary instead
inline constexpr int kLegacyJitOptLevel = -1;

// per-campaign log folder; runs are written into it as they complete
extern std::string gCampaignDir;

// maps a run's status ("OK"/"WARN"/"ERROR") onto its campaign subfolder
// ("ok"/"warn"/"fail")
const char *statusDirFor(const RunInfo &info);

// creates (or reuses) the campaign folder used by both pipelines
void createCampaignDir(const PipelineOptions &opts);

// helper function for multi mode, collects all .mlir files in the given folder
std::vector<std::string> collectMLIRFiles(const std::string &folder);

// picks a random .mlir file for a run; the per-run file RNG is derived only
// from seed and run index, so a fixed seed picks the same file every time
std::string pickInputFile(const PipelineOptions &opts,
                          const std::vector<std::string> &multiFiles,
                          int runIdx, int runSeed);

// per-run seed: fixed when --seed is given, otherwise derived
// deterministically from the run index so a campaign is reproducible
// without a fixed seed and never draws fresh entropy
int runSeedFor(const PipelineOptions &opts, int runIdx);

// parses a source file, capturing diagnostics in error
bool parseModuleFile(const std::string &file, mlir::MLIRContext &ctx,
                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                     std::string &error);

// parses a source module, clones it, and runs the metamorphic pass pipeline
// on the clone; used by both the full pipeline and emit mode
bool applyTransforms(MLIRSetup &setup, const std::string &inputFile,
                     mlir::OwningOpRef<mlir::ModuleOp> &originalModule,
                     mlir::OwningOpRef<mlir::ModuleOp> &transformedModule);

// Applies the jitter transform to a module: the pass runs with the fixed
// transform name and the run's own seed, so both sides of the comparison
// receive the same seeded RNG stream and the jitter density stays symmetric.
bool applyJitter(mlir::MLIRContext &ctx, mlir::ModuleOp module, int seed,
                 std::string &error);

// lowers a module to the LLVM dialect and translates it to LLVM IR,
// capturing diagnostics in error; IR dumps are only produced when requested
bool lowerAndTranslate(mlir::ModuleOp module, mlir::MLIRContext &mlirCtx,
                       llvm::LLVMContext &llvmCtx, const std::string &label,
                       std::string *loweredMLIR, std::string *llvmIR,
                       std::unique_ptr<llvm::Module> &llvmModule,
                       std::string &error);

} // namespace mlir_mracle
