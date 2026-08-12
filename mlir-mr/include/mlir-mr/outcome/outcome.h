#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace mlir_mr {

// outcome frequencies map for each output variable of a module
using OutcomeCounts = std::vector<std::map<int64_t, int>>;

// thread-count sweep per run; powers of two keep the sweep cheap
inline constexpr int kThreadCounts[] = {1, 2, 4, 8};

// adaptive execution: a small baseline, then chunked growth until the
// Good-Turing missing-mass estimate (N1/N) drops below the threshold or
// kMaxRuns is reached; per-source counts accumulate across runs. kMinRuns is
// sized so rare outcomes (~1/2000) have a real chance of being observed.
inline constexpr int kMinRuns = 2000;
inline constexpr int kCheckInterval = 100;
inline constexpr int kMaxRuns = 20000;
inline constexpr double kMissingMassThreshold = 0.005;

// Runs a pre-compiled callable numRuns times from numThreads concurrent
// callers and returns the outcome counts for each output variable.
OutcomeCounts executeCompiled(const std::function<std::vector<int64_t>()> &fn,
                              int numRuns, int numThreads);

// total executions reflected in the outcome counts
int totalExecutions(const OutcomeCounts &counts);

// merges src into dst, growing dst for extra output variables
void mergeCounts(OutcomeCounts &dst, const OutcomeCounts &src);

// Good-Turing missing-mass estimate N1/N: the fraction of observed values
// that occurred exactly once estimates the probability that the next run
// reveals a value not seen before. Returns the worst case across outputs so
// every output variable is sampled enough before stopping. When nothing was
// seen exactly once, a conservative 1/(N+1) floor keeps rare unseen outcomes
// from being declared saturated prematurely.
double missingMass(const OutcomeCounts &counts);

// Runs fn adaptively: a small baseline, then kCheckInterval-sized chunks
// while the Good-Turing missing-mass estimate stays at or above the
// threshold. Pre-existing counts (e.g. accumulated by earlier runs of the
// same source program) are carried forward instead of discarded.
OutcomeCounts executeCompiledAdaptive(
    const std::function<std::vector<int64_t>()> &fn, int numThreads,
    OutcomeCounts counts);

} // namespace mlir_mr
