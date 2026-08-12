#include "mlir-mr/oracle/oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

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

FisherVerdict classifyPValue(double p) {
    if (p < kPFail)
        return FisherVerdict::Fail;
    if (p < kPWarn)
        return FisherVerdict::Warn;
    return FisherVerdict::Pass;
}

namespace {

// log of k! for k in [0, n]; precomputed once per Fisher test so pmf
// evaluations become table lookups
struct LogFactorials {
    std::vector<double> lf;

    explicit LogFactorials(int n) : lf(static_cast<size_t>(n) + 1) {
        lf[0] = 0.0;
        for (int i = 1; i <= n; ++i)
            lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
    }

    double logC(int n, int k) const {
        return lf[n] - lf[k] - lf[n - k];
    }
};

// draws from Hypergeometric(N, K, n) by inverse transform over the pmf
int drawHypergeometric(const LogFactorials &lf, int N, int K, int n,
                       std::mt19937_64 &rng) {
    if (n == 0 || K == 0)
        return 0;
    if (K == N || n == N)
        return K;
    int lo = std::max(0, n - (N - K));
    int hi = std::min(n, K);
    double logDenom = lf.logC(N, n);
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double cdf = 0.0;
    for (int x = lo; x <= hi; ++x) {
        cdf += std::exp(lf.logC(K, x) + lf.logC(N - K, n - x) - logDenom);
        if (u < cdf)
            return x;
    }
    return hi;
}

// log probability of a row-0 allocation `a` for a 2xK table with column
// sums `c` and row totals n0/n1 (multivariate hypergeometric distribution)
double logTableProbability(const std::vector<int> &a, const std::vector<int> &c,
                           const LogFactorials &lf, int n0, int n1) {
    double p = lf.logC(n0 + n1, n0);
    for (size_t k = 0; k < c.size(); ++k)
        p += lf.logC(c[k], a[k]);
    return p;
}

// samples a table from the conditional (multivariate hypergeometric)
// distribution with the given margins by sequential draws
void sampleTable(std::vector<int> &a, const std::vector<int> &c, int n0,
                 const LogFactorials &lf, std::mt19937_64 &rng) {
    int row0Left = n0;
    int pool = 0;
    for (int ck : c)
        pool += ck;
    for (size_t k = 0; k + 1 < c.size(); ++k) {
        a[k] = (row0Left > 0 && c[k] > 0)
                   ? drawHypergeometric(lf, pool, c[k], row0Left, rng)
                   : 0;
        pool -= c[k];
        row0Left -= a[k];
    }
    a[c.size() - 1] = row0Left;
}

std::string formatP(double p) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", p);
    return std::string(buf);
}

std::string setFromCounts(const std::vector<int64_t> &cats,
                          const std::vector<int> &counts) {
    std::string s = "{";
    bool first = true;
    for (size_t i = 0; i < cats.size(); ++i) {
        if (counts[i] <= 0)
            continue;
        if (!first)
            s += ", ";
        first = false;
        s += std::to_string(cats[i]);
    }
    s += "}";
    return s;
}

std::string setFromMap(const std::map<int64_t, int> &m) {
    std::string s = "{";
    bool first = true;
    for (auto &[v, _] : m) {
        if (!first)
            s += ", ";
        first = false;
        s += std::to_string(v);
    }
    s += "}";
    return s;
}

std::string countsFor(const std::vector<int64_t> &cats,
                      const std::vector<int> &counts,
                      const std::vector<int64_t> &vals) {
    std::string s;
    bool first = true;
    for (auto v : vals) {
        int c = 0;
        for (size_t i = 0; i < cats.size(); ++i)
            if (cats[i] == v) {
                c = counts[i];
                break;
            }
        if (!first)
            s += ", ";
        first = false;
        s += std::to_string(v) + " (" + std::to_string(c) + ")";
    }
    return s;
}

} // namespace

FisherResult fisherTest(const std::map<int64_t, int> &source,
                        const std::map<int64_t, int> &transformed) {
    FisherResult fr;

    auto itS = source.begin(), itT = transformed.begin();
    while (itS != source.end() || itT != transformed.end()) {
        int64_t v;
        if (itT == transformed.end() ||
            (itS != source.end() && itS->first < itT->first)) {
            v = itS->first;
            ++itS;
        } else if (itS == source.end() || itT->first < itS->first) {
            v = itT->first;
            ++itT;
        } else {
            v = itS->first;
            ++itS;
            ++itT;
        }
        int sc = source.count(v) > 0 ? source.at(v) : 0;
        int tc = transformed.count(v) > 0 ? transformed.at(v) : 0;
        fr.categories.push_back(v);
        fr.sourceCounts.push_back(sc);
        fr.transformedCounts.push_back(tc);
        if (sc == 0)
            fr.novelOutcomes.push_back(v);
        if (tc == 0)
            fr.disappearedOutcomes.push_back(v);
    }

    int n0 = 0, n1 = 0;
    for (int c : fr.sourceCounts)
        n0 += c;
    for (int c : fr.transformedCounts)
        n1 += c;
    size_t K = fr.categories.size();

    // nothing to test: one side empty, or a single category cannot show
    // association between program and outcome
    if (n0 == 0 || n1 == 0 || K < 2)
        return fr;

    std::vector<int> c(K);
    for (size_t i = 0; i < K; ++i)
        c[i] = fr.sourceCounts[i] + fr.transformedCounts[i];
    LogFactorials lf(n0 + n1);
    double obsLogP = logTableProbability(fr.sourceCounts, c, lf, n0, n1);

    // deterministic seed derived from the table so results are reproducible
    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < K; ++i) {
        seed ^= static_cast<uint64_t>(fr.categories[i]) +
                0x9e3779b9ULL + (seed << 6) + (seed >> 2);
        seed ^= static_cast<uint64_t>(fr.sourceCounts[i]) +
                0x9e3779b9ULL + (seed << 6) + (seed >> 2);
        seed ^= static_cast<uint64_t>(fr.transformedCounts[i]) +
                0x9e3779b9ULL + (seed << 6) + (seed >> 2);
    }
    std::mt19937_64 rng(seed);

    std::vector<int> a(K);
    int extreme = 0;
    int sims = 0;
    double p = 1.0, pLo = 1.0, pHi = 1.0;

    while (sims < kMcMaxSimulations) {
        int batch = std::min(kMcBatchSize, kMcMaxSimulations - sims);
        for (int i = 0; i < batch; ++i) {
            sampleTable(a, c, n0, lf, rng);
            if (logTableProbability(a, c, lf, n0, n1) <= obsLogP)
                ++extreme;
        }
        sims += batch;
        p = static_cast<double>(extreme) / sims;

        // Wilson score interval at the configured confidence level
        double z = kMcConfidenceZ, z2 = z * z;
        double denom = 1.0 + z2 / sims;
        double center = (p + z2 / (2.0 * sims)) / denom;
        double margin = z * std::sqrt(p * (1.0 - p) / sims +
                                      z2 / (4.0 * sims * sims)) /
                        denom;
        pLo = std::max(0.0, center - margin);
        pHi = std::min(1.0, center + margin);

        // stop once the CI lies entirely within one verdict band
        if (pHi < kPFail || pLo >= kPWarn ||
            (pLo >= kPFail && pHi <= kPWarn))
            break;
    }

    fr.pValue = p;
    fr.pLow = pLo;
    fr.pHigh = pHi;
    fr.simulations = sims;
    return fr;
}

SequentialResult compareSequential(
    const std::function<std::vector<int64_t>()> &sourceFn,
    const std::function<std::vector<int64_t>()> &transformedFn,
    int numThreads, int maxRuns, int batchSize,
    const OutcomeCounts &resumeSrc, const OutcomeCounts &resumeTr,
    const std::function<void(const OutcomeCounts &, const OutcomeCounts &,
                             int, int)> &onBatch) {

    SequentialResult result;
    OutcomeCounts srcCounts = resumeSrc;
    OutcomeCounts trCounts = resumeTr;
    result.sourceRuns = totalExecutions(srcCounts);
    result.transformedRuns = totalExecutions(trCounts);

    int totalBatches = maxRuns / batchSize;
    int batchesDone = result.sourceRuns / batchSize;

    auto runFisher = [&]() {
        size_t numOutputs = std::max(srcCounts.size(), trCounts.size());
        result.variables.clear();
        result.variables.reserve(numOutputs);
        for (size_t v = 0; v < numOutputs; ++v) {
            std::map<int64_t, int> srcMap, trMap;
            if (v < srcCounts.size())
                srcMap = srcCounts[v];
            if (v < trCounts.size())
                trMap = trCounts[v];
            result.variables.push_back(fisherTest(srcMap, trMap));
        }
    };

    for (int b = batchesDone; b < totalBatches; ++b) {
        int catsBefore = 0;
        for (const auto &m : srcCounts)
            catsBefore += m.size();
        for (const auto &m : trCounts)
            catsBefore += m.size();

        mergeCounts(srcCounts, executeCompiled(sourceFn, batchSize, numThreads));
        mergeCounts(trCounts, executeCompiled(transformedFn, batchSize, numThreads));
        result.sourceRuns += batchSize;
        result.transformedRuns += batchSize;

        if (onBatch)
            onBatch(srcCounts, trCounts, result.sourceRuns,
                    result.transformedRuns);

        int catsAfter = 0;
        for (const auto &m : srcCounts)
            catsAfter += m.size();
        for (const auto &m : trCounts)
            catsAfter += m.size();
        bool newCategories = catsAfter > catsBefore;

        runFisher();

        // classify using the confidence interval so early verdicts are only
        // issued once the p-value is confidently inside a band
        bool anyFail = false, allPass = true;
        for (const auto &fr : result.variables) {
            if (fr.pHigh < kPFail) {
                anyFail = true;
                allPass = false;
            } else if (fr.pLow < kPWarn) {
                allPass = false;
            }
        }
        if (anyFail) {
            result.verdict = FisherVerdict::Fail;
            return result;
        }
        if (allPass && !newCategories && (b + 1) >= kMinBatchesForPass) {
            result.verdict = FisherVerdict::Pass;
            return result;
        }
    }

    // max runs exhausted without a confident early verdict: classify on the
    // point estimates
    runFisher();
    result.verdict = FisherVerdict::Pass;
    for (const auto &fr : result.variables) {
        FisherVerdict v = classifyPValue(fr.pValue);
        if (v == FisherVerdict::Fail) {
            result.verdict = FisherVerdict::Fail;
            return result;
        }
        if (v == FisherVerdict::Warn && result.verdict == FisherVerdict::Pass)
            result.verdict = FisherVerdict::Warn;
    }
    return result;
}

std::string formatFisherResult(const FisherResult &fr, size_t index,
                               size_t total, bool verbose) {
    std::string label =
        total == 1 ? "output" : "output " + std::to_string(index);
    std::string s = label + ": p=" + formatP(fr.pValue) + " (" +
                    std::to_string(fr.simulations) + " simulations)";
    if (verbose) {
        s += "\n  source outcome set: " +
             setFromCounts(fr.categories, fr.sourceCounts);
        s += "\n  transformed outcome set: " +
             setFromCounts(fr.categories, fr.transformedCounts);
    }
    if (!fr.novelOutcomes.empty())
        s += (verbose ? "\n  novel outcomes: " : "; novel outcomes: ") +
             countsFor(fr.categories, fr.transformedCounts, fr.novelOutcomes);
    if (!fr.disappearedOutcomes.empty())
        s += (verbose ? "\n  disappeared outcomes: " : "; disappeared outcomes: ") +
             countsFor(fr.categories, fr.sourceCounts, fr.disappearedOutcomes);
    return s;
}

CompareResult sequentialToCompareResult(const SequentialResult &seq,
                                        int numThreads, bool verbose) {
    if (seq.verdict == FisherVerdict::Pass)
        return {true, false, "Fisher p>=0.01"};

    std::string header =
        seq.verdict == FisherVerdict::Fail
            ? "=== SUMMARY: behavioural change detected (Fisher p<0.001) ===\n"
            : "=== SUMMARY: possible behavioural change (0.001<=p<0.01) ===\n";

    std::string body;
    for (size_t v = 0; v < seq.variables.size(); ++v) {
        if (!body.empty())
            body += "\n";
        body += formatFisherResult(seq.variables[v], v,
                                   seq.variables.size(), verbose);
    }

    bool ok = seq.verdict != FisherVerdict::Fail;
    return {ok, true, header + body};
}

CompareResult compareSingleThread(const OutcomeCounts &source,
                                  const OutcomeCounts &transformed,
                                  int runs) {
    if (source.size() != transformed.size())
        return {false, false,
                "result arity mismatch: source produces " +
                    std::to_string(source.size()) +
                    " outputs, transformed produces " +
                    std::to_string(transformed.size())};
    for (size_t v = 0; v < source.size(); ++v) {
        if (source[v] != transformed[v])
            return {false, false,
                    "non-deterministic difference: source produces " +
                        setFromMap(source[v]) +
                        ", transformed produces " +
                        setFromMap(transformed[v]) +
                        " over " + std::to_string(runs) + " runs each"};
        if (source[v].size() > 1)
            return {true, true,
                    "source is non-deterministic with 1 thread (" +
                        setFromMap(source[v]) +
                        ") - determinism assumption violated"};
    }
    return {true, false,
            "deterministic: identical single outcome over " +
                std::to_string(runs) + " runs"};
}

} // namespace mlir_mr
