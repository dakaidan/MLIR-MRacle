#pragma once

#include "conquer/core/types.h"

#include <map>
#include <optional>
#include <string>
#include <variant>

// Suppress warnings from nlohmann/json
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include <mlir/IR/BuiltinOps.h.inc>

#include <nlohmann/json.hpp>

namespace conquer {
using json = nlohmann::json;

/// Defines the action to be taken for a given layer.
enum class Action {
    Quantise, // Apply quantisation to the layer.
    Skip      // Do not quantise the layer, leaving it in its original precision.
};

/// Defines the numerical precision to be used for quantisation.
/// Any additional Precision types should be added to the json serialisation as well.
using Precision = DataType;

/// Defines the quantisation scheme (i.e., how the zero-point is handled).
enum class Scheme { Symmetric, Affine };

/// Defines the method used for calibrating the quantisation parameters (i.e., the scale and zero-point).
enum class CalibrationMethod { MinMax, Entropy, Percentile };

/// Defines the granularity of the quantisation.
enum class GranularityType { PerTensor, PerChannel };

// --- Granularity Configuration ---
struct PerTensorGranularity {};
struct PerChannelGranularity {
    int axis;
};
using Granularity = std::variant<PerTensorGranularity, PerChannelGranularity>;

/// Represents the configuration for quantising a single layer.
struct QuantisationConfig {
    Precision precision;
    Granularity granularity;
    Scheme scheme;
    CalibrationMethod calibration;

    /// Provides a default configuration for a quantised layer.
    QuantisationConfig()
        : precision(Precision::INT8), granularity(PerTensorGranularity{}), scheme(Scheme::Symmetric),
          calibration(CalibrationMethod::MinMax) {}
};

/// A placeholder for when a layer is to be skipped.
struct SkipConfig {};

/// Represents the configuration for a single layer, which can either be a `QuantisationConfig`
/// or a `SkipConfig`. In the context of the evolutionary algorithm, this is a "gene".
using LayerConf = std::variant<SkipConfig, QuantisationConfig>;

inline std::optional<QuantisationConfig> get_quant_config(const LayerConf &conf) {
    if (std::holds_alternative<QuantisationConfig>(conf)) {
        return std::get<QuantisationConfig>(conf);
    }
    return std::nullopt;
}

/// Represents the quantisation policy for a single layer, including separate configurations for weights and
/// activations.
struct Layer {
    LayerConf weight_policy;
    LayerConf activation_policy;
    bool squash_acc;
};

struct RawLayer {
    std::optional<QuantisationConfig> weight_policy;
    std::optional<QuantisationConfig> activation_policy;
    bool squash_acc;
};

/// Represents the global quantisation policy for the model.
struct GlobalPolicy {
    QuantisationConfig weight_policy;
    QuantisationConfig activation_policy;
};

/// The top-level data structure for a quantisation policy. It contains metadata, a global policy,
/// and a map of per-layer policies. In the context of the evolutionary algorithm, this is a "chromosome".
struct QuantisationPolicy {
    /// A map for storing arbitrary metadata about the policy (e.g., its fitness score, generation number, etc.).
    std::map<std::string, std::string> meta;
    /// The default quantisation policy to be applied to all layers that do not have a specific override.
    /// Either a `QuantisationConfig` or none.
    std::optional<GlobalPolicy> global_policy;
    /// A map of per-layer policies, where the key is the name of the operation.
    std::map<std::string, Layer> layers;

    [[nodiscard]] std::optional<RawLayer> get_config(const std::string &layer_name) const {
        if (const auto it = layers.find(layer_name); it != layers.end()) {
            return RawLayer{get_quant_config(it->second.weight_policy), get_quant_config(it->second.activation_policy), it->second.squash_acc};
        }
        if (global_policy.has_value()) {
            return RawLayer{get_quant_config(global_policy->weight_policy),
                            get_quant_config(global_policy->activation_policy),
                            false};
        }
        return std::nullopt;
    }

    [[nodiscard]] bool has_config(const std::string &layer_name) const {
        if (layers.contains(layer_name) || global_policy.has_value()) {
            if (const auto configOpt = get_config(layer_name)) {
                const auto &[weight_policy, activation_policy, _] = configOpt.value();
                const auto isSkip = [](const LayerConf &conf) { return std::holds_alternative<SkipConfig>(conf); };

                const bool w_skip = !weight_policy.has_value() || isSkip(weight_policy.value());
                const bool a_skip = !activation_policy.has_value() || isSkip(activation_policy.value());

                return !(w_skip && a_skip);
            }
        }
        return false;
    }
};

QuantisationPolicy parse_policy(const std::string &json_content);
std::string to_string(const QuantisationPolicy &policy, bool pretty = true);
std::string to_string(const Layer &layer, bool pretty = true);
QuantisationPolicy generate_default_policy(
    mlir::ModuleOp module,
    const std::optional<std::vector<Precision>>& allowed_precisions = std::nullopt);
void repair_policy(QuantisationPolicy &policy, mlir::ModuleOp module);

[[nodiscard]] std::vector<Layer> generate_mutations(
    std::string layer_entry,
    bool has_const_operand = true,
    const std::optional<std::vector<Precision>>& allowed_precisions = std::nullopt,
    const std::optional<std::vector<CalibrationMethod>>& allowed_calibrations = std::nullopt);
} // namespace conquer