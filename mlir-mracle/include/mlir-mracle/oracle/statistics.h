#pragma once

#include "mlir-mracle/context/context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace mlir_mracle {
namespace oracle_detail {

// p-value below which an outcome count is flagged as above the Poisson
// upper bound of the expected count
inline constexpr double kStrongPValue = 1e-6;

// regularised lower incomplete gamma P(a,x) = gamma(a,x)/Gamma(a): series for
// x < a+1, continued fraction for the upper tail otherwise
inline double lowerIncompleteGamma(double a, double x) {
    constexpr double kEps = 1e-15;
    constexpr int kItMax = 200;
    constexpr double kFpMin = 1e-300;
    const double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 0; n < kItMax; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * kEps)
                break;
        }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kItMax; ++i) {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kFpMin)
            d = kFpMin;
        c = b + an / c;
        if (std::fabs(c) < kFpMin)
            c = kFpMin;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps)
            break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

// P(X > k) for X ~ Poisson(lambda) == scipy.stats.poisson.sf(k, lambda); uses
// P(Poisson(lambda) >= a) = P(a, lambda) for integer a
inline double poissonSurvival(int64_t k, double lambda) {
    if (k < 0)
        return 1.0;
    if (lambda <= 0.0)
        return 0.0;
    return lowerIncompleteGamma(static_cast<double>(k) + 1.0, lambda);
}

// number of runs of `set` that produced `o` (0 if absent)
inline int64_t outcomeCount(const ObservedOutcomeSet &set,
                            const JointOutcome &o) {
    auto it = std::lower_bound(set.outcomes.begin(), set.outcomes.end(), o);
    if (it == set.outcomes.end() || *it != o)
        return 0;
    size_t idx = static_cast<size_t>(std::distance(set.outcomes.begin(), it));
    return idx < set.counts.size() ? set.counts[idx] : 0;
}

inline bool arityCompatible(const ObservedOutcomeSet &source,
                            const ObservedOutcomeSet &transformed) {
    return source.arityConsistent && transformed.arityConsistent &&
           source.arity == transformed.arity;
}

} // namespace oracle_detail
} // namespace mlir_mracle
