#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Dense>

namespace conquer {
struct CalibrationStats {
    // Absolute Min-Max (Exact for weights, running for activations)
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    // Exponential Moving Average (Meaningful for Activations only; defaults to absolute for weights)
    float emaMin = 0.0f;
    float emaMax = 0.0f;

    // Percentiles (0.1% and 99.9%)
    float pctMin = 0.0f;
    float pctMax = 0.0f;

    // KL-Divergence / Entropy clipping thresholds
    float klInt8Min = 0.0f;
    float klInt8Max = 0.0f;
    float klInt4Min = 0.0f;
    float klInt4Max = 0.0f;
};

class CalibrationCollector {
  public:
    explicit CalibrationCollector(const int numBins = 2048) : m_numBins(numBins), m_histogram(numBins, 0.0f) {}

    // --------------------------------------------------------------------
    // Bulk Processing used for Weights
    // Computes exact percentiles via sorting and builds a fixed histogram.
    // --------------------------------------------------------------------
    void processBulk(const std::vector<float> &values) {
        if (values.empty())
            return;

        auto [minIt, maxIt] = std::ranges::minmax_element(values);
        m_stats.min = *minIt;
        m_stats.max = *maxIt;
        m_stats.emaMin = m_stats.min;
        m_stats.emaMax = m_stats.max;
        m_totalElements = values.size();

        std::vector<float> sortedValues = values;
        std::ranges::sort(sortedValues);

        const auto lowerIdx = static_cast<size_t>(static_cast<double>(sortedValues.size()) * 0.001);
        auto upperIdx = static_cast<size_t>(static_cast<double>(sortedValues.size()) * 0.999);
        upperIdx = std::min(upperIdx, sortedValues.size() - 1);

        m_stats.pctMin = sortedValues[lowerIdx];
        m_stats.pctMax = sortedValues[upperIdx];

        m_absMax = std::max(std::abs(m_stats.min), std::abs(m_stats.max));
        if (m_absMax > 0.0f) {
            const float binWidth = m_absMax / static_cast<float>(m_numBins);
            for (const float val : values) {
                const int bin = std::clamp(static_cast<int>(std::abs(val) / binWidth), 0, m_numBins - 1);
                m_histogram[bin] += 1.0f;
            }
        }
    }

    // --------------------------------------------------------------------
    // Streaming Updates used for Activations
    // Computes EMA and uses a dynamic histogram to approximate percentiles/KL.
    // --------------------------------------------------------------------
    void updateStreaming(const std::vector<float> &values) {
        if (values.empty())
            return;

        auto [batchMinIt, batchMaxIt] = std::ranges::minmax_element(values);
        const float batchMin = *batchMinIt;
        const float batchMax = *batchMaxIt;

        m_stats.min = std::min(m_stats.min, batchMin);
        m_stats.max = std::max(m_stats.max, batchMax);

        if (m_totalElements == 0) {
            m_stats.emaMin = batchMin;
            m_stats.emaMax = batchMax;
            m_absMax = std::max(std::abs(batchMin), std::abs(batchMax)) + 1e-5f; // Initial histogram range
        } else {
            constexpr float momentum = 0.01f;
            m_stats.emaMin = m_stats.emaMin - momentum * (m_stats.emaMin - batchMin);
            m_stats.emaMax = m_stats.emaMax - momentum * (m_stats.emaMax - batchMax);
        }

        m_totalElements += values.size();

        for (const float val : values) {
            const float absVal = std::abs(val);
            while (absVal >= m_absMax) {
                compressHistogram();
            }
            const float binWidth = m_absMax / static_cast<float>(m_numBins);
            const int bin = std::clamp(static_cast<int>(absVal / binWidth), 0, m_numBins - 1);
            m_histogram[bin] += 1.0f;
        }
    }

    CalibrationStats getStats() {
        if (m_totalElements == 0)
            return m_stats;

        // If streaming was used, we need to approximate percentiles from the histogram
        if (m_stats.pctMin == 0.0f && m_stats.pctMax == 0.0f && (m_stats.min != 0.0f || m_stats.max != 0.0f)) {
            approximatePercentiles();
        }

        if (m_absMax > 0.0f) {
            const float klInt8Threshold = findOptimalKLThreshold(128);
            m_stats.klInt8Min = -klInt8Threshold;
            m_stats.klInt8Max = klInt8Threshold;

            const float klInt4Threshold = findOptimalKLThreshold(8);
            m_stats.klInt4Min = -klInt4Threshold;
            m_stats.klInt4Max = klInt4Threshold;
        }

        return m_stats;
    }

  private:
    CalibrationStats m_stats;
    size_t m_totalElements = 0;

    int m_numBins;
    float m_absMax = 0.0f;
    std::vector<float> m_histogram;

    void compressHistogram() {
        m_absMax *= 2.0f;
        for (int i = 0; i < m_numBins / 2; ++i) {
            m_histogram[i] = m_histogram[2 * i] + m_histogram[2 * i + 1];
        }
        std::fill(m_histogram.begin() + m_numBins / 2, m_histogram.end(), 0.0f);
    }

    void approximatePercentiles() {
        const float targetCount = static_cast<float>(m_totalElements) * 0.999f;
        float currentSum = 0.0f;
        const float binWidth = m_absMax / static_cast<float>(m_numBins);

        for (int i = 0; i < m_numBins; ++i) {
            currentSum += m_histogram[i];
            if (currentSum >= targetCount) {
                const float threshold = static_cast<float>(i + 1) * binWidth;
                // Because histogram is absolute values, we bound symmetrically
                m_stats.pctMin = std::max(m_stats.min, -threshold);
                m_stats.pctMax = std::min(m_stats.max, threshold);
                return;
            }
        }
    }

    [[nodiscard]] float findOptimalKLThreshold(const int targetBins) const {
        float minKlDivergence = std::numeric_limits<float>::max();
        int optimalThresholdBin = targetBins;
        const float binWidth = m_absMax / static_cast<float>(m_numBins);

        for (int thresholdBin = targetBins; thresholdBin < m_numBins; ++thresholdBin) {
            std::vector referenceDist(m_histogram.begin(), m_histogram.begin() + thresholdBin);

            // Clip outliers
            for (int i = thresholdBin; i < m_numBins; ++i) {
                referenceDist.back() += m_histogram[i];
            }

            std::vector quantDist(thresholdBin, 0.0f);
            const float numElementsPerQuantBin = static_cast<float>(thresholdBin) / static_cast<float>(targetBins);

            for (int i = 0; i < targetBins; ++i) {
                float sum = 0.0f;
                const int start = static_cast<int>(static_cast<float>(i) * numElementsPerQuantBin);
                const int end = static_cast<int>(static_cast<float>(i + 1) * numElementsPerQuantBin);
                for (int j = start; j < end; ++j)
                    sum += referenceDist[j];

                if (sum > 0.0f) {
                    for (int j = start; j < end; ++j) {
                        if (referenceDist[j] > 0.0f) {
                            quantDist[j] = sum / static_cast<float>(end - start);
                        }
                    }
                }
            }

            float klDivergence = 0.0f;
            for (int i = 0; i < thresholdBin; ++i) {
                if (referenceDist[i] > 0.0f && quantDist[i] > 0.0f) {
                    klDivergence += referenceDist[i] * std::log(referenceDist[i] / quantDist[i]);
                }
            }

            if (klDivergence < minKlDivergence) {
                minKlDivergence = klDivergence;
                optimalThresholdBin = thresholdBin;
            }
        }
        return (static_cast<float>(optimalThresholdBin) + 0.5f) * binWidth;
    }
};

struct SensitivityStats {
    // Normalized Shannon Entropy of Eigenvalues.
    // Bounded [0.0, 1.0].
    // 1.0 = highly sensitive (fragile), 0.0 = highly redundant (robust).
    float entropySensitivity = 0.0f;
};

class SensitivityCollector {
public:
    void init(const int64_t channels) {
        num_channels = channels;
        mean.assign(channels, 0.0);
        S.assign(channels * channels, 0.0);
    }

    // Pass in a single feature vector of size C
    void update(const float* feature_vector) {
        if (num_channels == 0) return;

        num_samples++;
        std::vector<double> delta(num_channels, 0.0);

        // Calculate Delta and update Mean
        for (int64_t c = 0; c < num_channels; ++c) {
            delta[c] = feature_vector[c] - mean[c];
            mean[c] += delta[c] / num_samples;
        }

        // Update unnormalized covariance matrix S
        for (int64_t i = 0; i < num_channels; ++i) {
            double diff = feature_vector[i] - mean[i];
            for (int64_t j = 0; j < num_channels; ++j) {
                // S = S + delta * (x - mean)^T
                S[i * num_channels + j] += delta[i] * diff;
            }
        }
    }

    // Helper to process a whole batched tensor
    void updateStreaming(const std::vector<float>& activations, const int64_t channels) {
        if (num_channels == 0) init(channels);

        // Assuming data is structured as N * H * W feature vectors of size C.
        // NOTE: depends on your exact layout (NCHW vs NHWC).
        // This assumes NHWC for contiguous C vectors.
        const int64_t total_elements = activations.size();
        const int64_t num_vectors = total_elements / channels;

        for (int64_t i = 0; i < num_vectors; ++i) {
            update(&activations[i * channels]);
        }
    }

    [[nodiscard]] std::vector<double> getCovarianceMatrix() const {
        std::vector<double> cov(num_channels * num_channels, 0.0);
        if (num_samples < 2) return cov;

        for (size_t i = 0; i < S.size(); ++i) {
            cov[i] = S[i] / (num_samples - 1);
        }
        return cov;
    }

    [[nodiscard]] SensitivityStats getStats() const {
        SensitivityStats stats;
        if (num_samples < 2 || num_channels <= 1) return stats;

        std::vector<double> cov_data = getCovarianceMatrix();

        Eigen::Map<Eigen::MatrixXd> cov(cov_data.data(), num_channels, num_channels);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();

        double sum_abs_eig = eigenvalues.cwiseAbs().sum();
        if (sum_abs_eig == 0.0) return stats;

        double entropy = 0.0;
        for (int i = 0; i < num_channels; ++i) {
            double prob = std::abs(eigenvalues[i]) / sum_abs_eig;
            if (prob > 1e-12) {
                entropy -= prob * std::log(prob);
            }
        }

        // Normalize by max possible entropy ( ln(C) ) to bound between 0.0 and 1.0
        double max_entropy = std::log(static_cast<double>(num_channels));
        stats.entropySensitivity = static_cast<float>(entropy / max_entropy);

        return stats;
    }

private:
    int64_t num_samples = 0;
    int64_t num_channels = 0;
    std::vector<double> mean;
    std::vector<double> S;    // Flattened C x C matrix
};
} // namespace conquer