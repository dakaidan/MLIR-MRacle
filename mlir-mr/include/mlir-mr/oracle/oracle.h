#pragma once

#include "mlir-mr/context/context.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mlir_mr {

// outcome frequencies map for each output variable of a module
using OutcomeCounts = std::vector<std::map<int64_t, int>>;

// thread-count sweep per run; the 2-thread primary test runs first
inline constexpr int kThreadCounts[] = {2, 1, 4, 8};

// maximum runs per (source | transformed) program at each thread level
inline constexpr int kRunsPrimary   = 50000; // 2 threads
inline constexpr int kRunsSecondary = 20000; // 4 and 8 threads
inline constexpr int kRunsSingle    = 1000;  // 1 thread (determinism check)

// sequential batch sizes for execution
inline constexpr int kBatchPrimary   = 5000;
inline constexpr int kBatchSecondary = 2000;

// p-value classification thresholds
inline constexpr double kPFail = 0.001;
inline constexpr double kPWarn = 0.01;

// Monte Carlo parameters for Fisher's exact test
inline constexpr int kMcBatchSize = 1000;
inline constexpr int kMcMaxSimulations = 100000;
inline constexpr double kMcConfidenceZ = 2.576; // 99% two-sided CI

// minimum batches before an early pass verdict is allowed, so rare outcomes
// get a chance to appear before the sequential test terminates
inline constexpr int kMinBatchesForPass = 2;

// Runs a pre-compiled callable numRuns times from numThreads concurrent
// callers and returns the outcome counts for each output variable.
OutcomeCounts executeCompiled(const std::function<std::vector<int64_t>()> &fn,
                              int numRuns, int numThreads);

// total executions reflected in the outcome counts
int totalExecutions(const OutcomeCounts &counts);

// merges src into dst, growing dst for extra output variables
void mergeCounts(OutcomeCounts &dst, const OutcomeCounts &src);

// classification of a p-value against the fail/warn thresholds
FisherVerdict classifyPValue(double p);

// 1-thread determinism check: with no concurrency both programs must be
// deterministic and agree; a multi-valued outcome set means the program is
// non-deterministic even single-threaded
CompareResult compareSingleThread(const OutcomeCounts &source,
                                  const OutcomeCounts &transformed,
                                  int runs);

// Freeman-Halton extension of Fisher's exact test for a single output
// variable. Builds a 2xK contingency table (source vs transformed, one
// column per distinct outcome value) and estimates the two-sided p-value
// with adaptive Monte Carlo simulation, stopping once the 99% confidence
// interval lies entirely within one classification band.
FisherResult fisherTest(const std::map<int64_t, int> &source,
                        const std::map<int64_t, int> &transformed);

// Runs both modules in lockstep batches of batchSize and tests after each
// batch. Stops early when every output variable is confidently a pass (p
// above kPWarn by confidence interval, no new categories this batch, at
// least kMinBatchesForPass batches done) or any output variable is
// confidently a fail (p below kPFail by confidence interval). Exhausts
// maxRuns otherwise. resumeSrc/resumeTr carry forward accumulated counts
// from a checkpointed partial run; onBatch is invoked after every batch
// with the accumulated counts so callers can persist a checkpoint.
SequentialResult compareSequential(
    const std::function<std::vector<int64_t>()> &sourceFn,
    const std::function<std::vector<int64_t>()> &transformedFn,
    int numThreads, int maxRuns, int batchSize,
    const OutcomeCounts &resumeSrc, const OutcomeCounts &resumeTr,
    const std::function<void(const OutcomeCounts &, const OutcomeCounts &,
                             int, int)> &onBatch = {});

// renders a Fisher result (outcome sets, p-value, novel/disappeared values)
std::string formatFisherResult(const FisherResult &fr, size_t index,
                               size_t total, bool verbose);

// renders a SequentialResult as a CompareResult for ThreadGroupResult
CompareResult sequentialToCompareResult(const SequentialResult &seq,
                                        int numThreads, bool verbose);

} // namespace mlir_mr
