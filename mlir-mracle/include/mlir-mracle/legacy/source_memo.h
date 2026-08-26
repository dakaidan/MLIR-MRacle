#pragma once

#include "mlir-mracle/io/cache.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir_mracle {

// a baseline cache write deferred until the run verdict is known: normally
// only OK runs commit, but a run that verified a warn (extra source data
// merged into the baseline) commits as well, so the grown baseline is
// persisted whether the warn resolved or was confirmed
struct PendingBaseline {
    std::string sourceHash;
    std::string baseKey;
    int reps = 0;
    ObservedOutcomeSet set;
};

// process-lifetime memoisation of source baselines: the compiled source
// binaries and the outcome sets collected from them are kept alive for the
// whole pipeline loop, so repeated runs of the same file reuse them instead
// of recompiling or re-executing. Nothing is written to disk, so no state
// survives between processes.
struct SourceMemo {
    std::string sourceMLIR;
    std::string sourceHash;
    std::string sourceJitLLVM;
    std::unique_ptr<llvm::LLVMContext> llvmCtx;
    std::function<std::vector<int64_t>()> plain;
    std::unique_ptr<llvm::Module> tsanModule; // consumed on first TSan compile
    std::function<std::vector<int64_t>()> tsan;
    std::string tsanError;
    std::map<std::string, ObservedOutcomeSet> baselines; // key "t[:tsan]"
    int warnCount = 0; // warns of this file; the first few are verified
};

// process-lifetime memoisation store, keyed by file path
extern std::map<std::string, SourceMemo> gSourceMemo;

// invalidates a baseline entry (memory and disk) once the per-run retest cap
// is reached with outcomes still missing: the entry can no longer learn, so
// the next run of this file recollects from scratch and can take in new
// possible values
void clearBaseline(SourceMemo &memo, const std::string &baseKey, int reps);

// lowers and compiles the source module once, caching the plain binary (and
// holding the module for an on-demand TSan compile) so later runs of the same
// file skip recompilation; the lowered bitcode is also persisted so later
// processes of the same source skip lowering and translation entirely
bool memoizeSource(mlir::MLIRContext &mlirCtx, mlir::ModuleOp module,
                   const std::string &sourceMLIR, SourceMemo &memo,
                   std::string &error);

// returns the TSan-instrumented source binary, compiling it on first use and
// keeping it alive for later runs of the same file
std::function<std::vector<int64_t>()> sourceTsanBinary(SourceMemo &memo,
                                                       std::string &error);

} // namespace mlir_mracle
