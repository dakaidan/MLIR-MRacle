#include "mlir-mr/outcome/outcome.h"

#include <algorithm>
#include <mutex>
#include <thread>

namespace mlir_mr {

OutcomeCounts executeCompiled(const std::function<std::vector<int64_t>()> &fn,
                              int numRuns, int numThreads) {
    OutcomeCounts counts;
    std::mutex countsMutex;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    int base = numRuns / numThreads;
    int extra = numRuns % numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int runs = base + (t < extra ? 1 : 0);
        threads.emplace_back([&, runs]() {
            for (int i = 0; i < runs; ++i) {
                auto results = fn();
                std::lock_guard<std::mutex> lock(countsMutex);
                if (counts.size() < results.size())
                    counts.resize(results.size());
                for (size_t r = 0; r < results.size(); ++r)
                    ++counts[r][results[r]];
            }
        });
    }
    for (auto &th : threads)
        th.join();

    return counts;
}

int totalExecutions(const OutcomeCounts &counts) {
    int total = 0;
    for (const auto &m : counts)
        for (const auto &[_, c] : m)
            total += c;
    return total;
}

void mergeCounts(OutcomeCounts &dst, const OutcomeCounts &src) {
    if (dst.size() < src.size())
        dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        for (const auto &[val, c] : src[i])
            dst[i][val] += c;
}

double missingMass(const OutcomeCounts &counts) {
    double worst = 0.0;
    for (const auto &m : counts) {
        int singletons = 0, total = 0;
        for (const auto &[_, c] : m) {
            total += c;
            if (c == 1)
                ++singletons;
        }
        if (total > 0) {
            double estimate = singletons > 0
                ? static_cast<double>(singletons) / total
                : 1.0 / (total + 1.0);
            worst = std::max(worst, estimate);
        }
    }
    return worst;
}

OutcomeCounts executeCompiledAdaptive(
    const std::function<std::vector<int64_t>()> &fn, int numThreads,
    OutcomeCounts counts) {
    if (totalExecutions(counts) == 0)
        counts = executeCompiled(fn, kMinRuns, numThreads);
    while (totalExecutions(counts) < kMaxRuns &&
           missingMass(counts) >= kMissingMassThreshold)
        mergeCounts(counts, executeCompiled(fn, kCheckInterval, numThreads));
    return counts;
}

} // namespace mlir_mr
