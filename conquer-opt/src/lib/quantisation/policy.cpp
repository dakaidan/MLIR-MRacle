#include "conquer/quantisation/policy.h"
#include "conquer/core/logging.h"
#include "conquer/core/config.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/Support/Debug.h>

// Suppress warnings from nlohmann/json scoped entirely to this translation unit
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include <nlohmann/json.hpp>
#pragma clang diagnostic pop

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-policy"

namespace conquer {
using json = nlohmann::json;

NLOHMANN_JSON_SERIALIZE_ENUM(Action, {{Action::Quantise, "quantise"}, {Action::Skip, "skip"}})
NLOHMANN_JSON_SERIALIZE_ENUM(Precision, {
                                        // {Precision::UINT4, "uint4"},
                                         {Precision::INT4, "int4"},
                                         // {Precision::UINT8, "uint8"},
                                         {Precision::INT8, "int8"},
                                         {Precision::INT16, "int16"},
                                         {Precision::INT32, "int32"},
                                         {Precision::FP32, "fp32"},
                                         {Precision::BF16, "bf16"},
                                         {Precision::FP16, "fp16"},
                                         {Precision::FP8_E4M3, "fp8_e4m3"},
                                         {Precision::FP8_E5M2, "fp8_e5m2"}})
NLOHMANN_JSON_SERIALIZE_ENUM(Scheme, {{Scheme::Symmetric, "symmetric"}, {Scheme::Affine, "affine"}})
NLOHMANN_JSON_SERIALIZE_ENUM(CalibrationMethod, {{CalibrationMethod::MinMax, "min_max"},
                                                 {CalibrationMethod::Entropy, "entropy"},
                                                 {CalibrationMethod::Percentile, "percentile"}})
NLOHMANN_JSON_SERIALIZE_ENUM(GranularityType,
                             {{GranularityType::PerTensor, "per_tensor"}, {GranularityType::PerChannel, "per_channel"}})

void to_json(json &j, const QuantisationConfig &cfg);
void from_json(const json &j, QuantisationConfig &cfg);
void to_json(json &j, const LayerConf &conf);
void from_json(const json &j, LayerConf &conf);
void to_json(json &j, const Layer &layer);
void from_json(const json &j, Layer &layer);
void to_json(json &j, const GlobalPolicy &gp);
void from_json(const json &j, GlobalPolicy &gp);
void to_json(json &j, const QuantisationPolicy &p);
void from_json(const json &j, QuantisationPolicy &p);

void to_json(json &j, const QuantisationConfig &cfg) {
    j["precision"] = cfg.precision;
    j["scheme"] = cfg.scheme;
    j["calibration"] = cfg.calibration;

    std::visit(
        [&j]<typename T0>(T0 &&arg) {
            using T = std::decay_t<T0>;
            if constexpr (std::is_same_v<T, PerTensorGranularity>) {
                j["granularity"] = "per_tensor";
            } else if constexpr (std::is_same_v<T, PerChannelGranularity>) {
                // j["granularity"] = {{"type", "per_channel"}, {"axis", arg.axis}};
                j["granularity"] = "per_channel";
            }
        },
        cfg.granularity);
}

void from_json(const json &j, QuantisationConfig &cfg) {
    if (j.contains("precision"))
        j.at("precision").get_to(cfg.precision);
    if (j.contains("scheme"))
        j.at("scheme").get_to(cfg.scheme);
    if (j.contains("calibration"))
        j.at("calibration").get_to(cfg.calibration);

    if (j.contains("granularity")) {
        if (j["granularity"].is_string()) {
            if (j["granularity"] == "per_tensor") {
                cfg.granularity = PerTensorGranularity{};
            } else if (j["granularity"] == "per_channel") {
                cfg.granularity = PerChannelGranularity{-1};
            }
        } else {
            cfg.granularity = PerChannelGranularity{j["granularity"].at("axis").get<int>()};
        }
    }
}

void to_json(json &j, const LayerConf &conf) {
    std::visit(
        [&j]<typename T0>(T0 &&arg) {
            using T = std::decay_t<T0>;
            if constexpr (std::is_same_v<T, SkipConfig>) {
                j = {{"action", "skip"}};
            } else if constexpr (std::is_same_v<T, QuantisationConfig>) {
                j = arg; // This calls the QuantisationConfig to_json
                j["action"] = "quantise";
            }
        },
        conf);
}

void from_json(const json &j, LayerConf &conf) {
    if (j.at("action").get<Action>() == Action::Skip) {
        conf = SkipConfig{};
    } else {
        conf = j.get<QuantisationConfig>();
    }
}

void to_json(json &j, const Layer &layer) {
    j["weight_policy"] = layer.weight_policy;
    j["activation_policy"] = layer.activation_policy;
    j["squash_acc"] = layer.squash_acc;
}

void from_json(const json &j, Layer &layer) {
    if (j.contains("weight_policy")) {
        j.at("weight_policy").get_to(layer.weight_policy);
    }
    if (j.contains("activation_policy")) {
        j.at("activation_policy").get_to(layer.activation_policy);
    }
    if (j.contains("squash_acc"))
        j.at("squash_acc").get_to(layer.squash_acc);
    else
        layer.squash_acc = false;
}

void to_json(json &j, const GlobalPolicy &gp) {
    j["weight_policy"] = gp.weight_policy;
    j["activation_policy"] = gp.activation_policy;
}

void from_json(const json &j, GlobalPolicy &gp) {
    if (j.contains("weight_policy")) {
        j.at("weight_policy").get_to(gp.weight_policy);
    }
    if (j.contains("activation_policy")) {
        j.at("activation_policy").get_to(gp.activation_policy);
    }
}

void to_json(json &j, const QuantisationPolicy &p) {
    j["meta"] = p.meta;
    if (p.global_policy.has_value()) {
        j["global_policy"] = p.global_policy.value();
    }
    j["layers"] = p.layers;
}

void from_json(const json &j, QuantisationPolicy &p) {
    j.at("meta").get_to(p.meta);
    if (j.contains("global_policy")) {
        GlobalPolicy gp;
        j.at("global_policy").get_to(gp);
        p.global_policy = gp;
    }
    j.at("layers").get_to(p.layers);
}

QuantisationPolicy parse_policy(const std::string &json_content) {
    const auto j = json::parse(json_content);
    return j.get<QuantisationPolicy>();
}

std::string to_string(const QuantisationPolicy &policy, const bool pretty) {
    const json j = policy;
    return pretty ? j.dump(4) : j.dump();
}

std::string to_string(const Layer &layer, const bool pretty) {
    const json j = layer;
    return pretty ? j.dump(4) : j.dump();
}

QuantisationPolicy generate_default_policy(mlir::ModuleOp module, const std::optional<std::vector<Precision>>& allowed_precisions) {
    L_INFO("Generating default quantisation policy for module.");
    QuantisationPolicy policy;
    policy.meta["generated_by"] = "conquer";

    Precision default_prec = Precision::FP32;

    policy.meta["description"] = "Default quantisation policy generated by Conquer. Base precision: " + json(default_prec).get<std::string>();

    QuantisationConfig default_config;
    default_config.precision = default_prec;

    module.walk([&](mlir::Operation *op) {
        const auto nameAttr = op->getAttrOfType<mlir::StringAttr>("conquer.name");
        if (!nameAttr) {
            return;
        }

        const std::string name = nameAttr.getValue().str();
        policy.layers[name] = Layer{default_config, default_config, false};
        L_TRACE("Added default policy for layer: " << name);
    });

    repair_policy(policy, module);

    return policy;
}

void repair_policy(QuantisationPolicy &policy, mlir::ModuleOp module) {
    L_INFO("Repairing quantisation policy to enforce constraints...");

    auto is_integer = [](const Precision p) {
        return p == Precision::INT4 || /*p == Precision::UINT4 ||*/ p == Precision::INT8 || /*p == Precision::UINT8 ||*/
               p == Precision::INT16 || p == Precision::INT32;
    };

    auto is_float_quant = [](const Precision p) {
        return p == Precision::FP8_E4M3 || p == Precision::FP8_E5M2 || p == Precision::BF16 || p == Precision::FP16;
    };

    std::map<std::string, bool> is_non_negative;
    module.walk([&](mlir::Operation *op) {
        const auto nameAttr = op->getAttrOfType<mlir::StringAttr>("conquer.name");
        const auto minAttr = op->getAttrOfType<mlir::FloatAttr>("activation_stats.min_max.min");

        if (nameAttr && minAttr) {
            const std::string name = nameAttr.getValue().str();
            const float min_val = minAttr.getValue().convertToFloat();

            if (!is_non_negative.contains(name)) {
                is_non_negative[name] = true;
            }

            if (min_val < 0.0f) {
                is_non_negative[name] = false;
            }
        }
    });

    for (auto &[name, layer] : policy.layers) {
        if (auto *w_cfg = std::get_if<QuantisationConfig>(&layer.weight_policy)) {
            if (is_float_quant(w_cfg->precision)) {
                if (w_cfg->scheme != Scheme::Symmetric) {
                    w_cfg->scheme = Scheme::Symmetric;
                    L_DEBUG("  [Repair] " << name
                               << " weights: Float formats do not support zero-points. Forced to Symmetric.");
                }
            } else if (is_integer(w_cfg->precision)) {
                if (w_cfg->scheme != Scheme::Symmetric) {
                    w_cfg->scheme = Scheme::Symmetric;
                    L_DEBUG("  [Repair] " << name
                               << " weights: Asymmetric INT scheme is hostile to hardware. Forced to Symmetric.");
                }
            } else if (w_cfg->precision == Precision::FP32) {
                layer.weight_policy = SkipConfig{};
            }
        }

        if (auto *a_cfg = std::get_if<QuantisationConfig>(&layer.activation_policy)) {

            if (std::holds_alternative<PerChannelGranularity>(a_cfg->granularity)) {
                a_cfg->granularity = PerTensorGranularity{};
                L_DEBUG("  [Repair] " << name
                                        << " activations: PerChannel requested. Forced to PerTensor.");
            }

            if (is_float_quant(a_cfg->precision)) {
                if (a_cfg->scheme != Scheme::Symmetric) {
                    a_cfg->scheme = Scheme::Symmetric;
                    L_DEBUG("  [Repair] " << name
                               << " activations: Float formats do not support zero-points. Forced to Symmetric.");
                }
            } else if (is_integer(a_cfg->precision)) {
                if (is_non_negative[name] && a_cfg->scheme != Scheme::Affine) {
                    a_cfg->scheme = Scheme::Affine;
                    L_DEBUG("  [Repair] " << name
                                            << " activations: Data is strictly non-negative (e.g. ReLU). Snapped to "
                                               "Affine to save precision.");
                }
            } else if (a_cfg->precision == Precision::FP32) {
                layer.activation_policy = SkipConfig{};
            }
        }

        const bool weights_skipped = std::holds_alternative<SkipConfig>(layer.weight_policy);
        const bool acts_skipped = std::holds_alternative<SkipConfig>(layer.activation_policy);

        if (weights_skipped && !acts_skipped) {
            layer.activation_policy = SkipConfig{};
            L_DEBUG("  [Repair] " << name
                       << ": Weights were skipped but activations were quantised. Snapped activations to Skip.");
        }
    }

    L_INFO("Policy repair complete.");
}

// Generates valid weight configurations for a given precision.
// Enforces TOSA and hardware constraints: weights are always Symmetric.
std::vector<LayerConf> generate_weight_configs(Precision p, bool allow_per_channel) {
    std::vector<LayerConf> configs;

    QuantisationConfig pt_config;
    pt_config.precision = p;
    pt_config.granularity = PerTensorGranularity{};
    pt_config.scheme = Scheme::Symmetric;
    pt_config.calibration = CalibrationMethod::MinMax;
    configs.emplace_back(pt_config);

    if (allow_per_channel) {
        QuantisationConfig pc_config = pt_config;
        pc_config.granularity = PerChannelGranularity{-1}; // Axis resolved downstream
        configs.emplace_back(pc_config);
    }

    return configs;
}

// Generates valid activation configurations for a given precision.
// Enforces TOSA constraints: Activations are Per-Tensor. Affine is
// reserved for INT8 to handle asymmetric ranges
std::vector<LayerConf> generate_act_configs(Precision p) {
    std::vector<LayerConf> configs;

    QuantisationConfig config;
    config.precision = p;
    config.granularity = PerTensorGranularity{};
    config.calibration = CalibrationMethod::MinMax;
    config.scheme = Scheme::Symmetric;
    configs.emplace_back(config);

    if (p == Precision::INT8) {
        QuantisationConfig affine_config = config;
        affine_config.scheme = Scheme::Affine;
        configs.emplace_back(affine_config);
    }

    return configs;
}

/**
 * @brief Generates all valid quantisation policy mutations for a given layer,
 * filtered by the physical capabilities of the target hardware.
 *
 * @param layer_entry The layer name and its current base policy.
 * @param has_const_operand True if the graph walker identified a tosa.const feeding into this operation.
 * @param allowed_precisions The cached list of natively supported precisions from the HardwareProfile.
 */
std::vector<Layer> generate_mutations(std::string layer_entry,
                                      bool has_const_operand,
                                      const std::optional<std::vector<Precision>>& allowed_precisions,
                                      const std::optional<std::vector<CalibrationMethod>>& allowed_calibrations) {
    std::vector<Layer> mutations;

    // Clean up the layer name (e.g., "tosa_conv2d_0" -> "conv2d")
    if (layer_entry.starts_with("tosa_")) layer_entry = layer_entry.substr(5);

    size_t last_underscore = layer_entry.find_last_of('_');
    if (last_underscore != std::string::npos &&
        last_underscore + 1 < layer_entry.size() &&
        std::isdigit(layer_entry[last_underscore + 1])) {
        layer_entry = layer_entry.substr(0, last_underscore);
    }

    bool supports_squash_acc = false;
    bool allow_per_channel_weights = false;
    bool has_weights = false;

    struct PrecPair {
        std::optional<Precision> act;
        std::optional<Precision> weight;
    };
    std::vector<PrecPair> valid_pairs;

    // --- HARDWARE FILTER LAMBDA ---
    // Only adds the precision pair to the search space if the hardware natively supports it.
    auto add_pair = [&](std::optional<Precision> act, std::optional<Precision> weight) {
        auto is_supported = [&](std::optional<Precision> p) {
            if (!p.has_value()) return true; // SkipConfig (FP32 baseline) is always supported
            if (!allowed_precisions.has_value()) return true; // No hardware profile provided = permit all
            if (allowed_precisions.value().empty()) return true;
            return std::ranges::find(*allowed_precisions, p.value()) != allowed_precisions->end();
        };

        // Both the activation AND the weight precision must be supported by the hardware
        if (is_supported(act) && is_supported(weight)) {
            valid_pairs.push_back({act, weight});
        }
    };

    // Every layer is allowed to remain completely unquantised (FP32 baseline / SkipConfig)
    add_pair(std::nullopt, std::nullopt);

    // --------------------------------------------------------------------
    // MATRIX & CONVOLUTION OPERATIONS
    // --------------------------------------------------------------------
    if (layer_entry == "conv2d" || layer_entry == "conv3d" ||
        layer_entry == "depthwise_conv2d" || layer_entry == "transpose_conv2d") {

        // CONVS: Deep accumulators, true weights, per-channel allowed.
        supports_squash_acc = true;
        has_weights = true;
        allow_per_channel_weights = true;

        // TOSA Conv Supported Combinations (Act, Weight)
        add_pair(Precision::FP16, Precision::FP16);
        add_pair(Precision::INT8, Precision::INT8);
        add_pair(Precision::BF16, Precision::BF16);
        add_pair(Precision::FP8_E4M3, Precision::FP8_E4M3);
        add_pair(Precision::FP8_E5M2, Precision::FP8_E5M2);
        if (IS_I16_ACTIVATION_SUPPORTED) {
                    add_pair(Precision::INT16, Precision::INT8);
        }
        if (IS_I4_WEIGHT_COMPUTE_SUPPORTED) {
                    add_pair(Precision::INT8, Precision::INT4);
        }

    } else if (layer_entry == "matmul") {

        // MATMUL: Deep accumulators, true weights, per-channel allowed.
        // Requires both inputs to be the exact same type.
        supports_squash_acc = true;
        has_weights = true;
        allow_per_channel_weights = true;

        add_pair(Precision::FP16, Precision::FP16);
        add_pair(Precision::INT8, Precision::INT8);
        add_pair(Precision::BF16, Precision::BF16);
        add_pair(Precision::FP8_E4M3, Precision::FP8_E4M3);
        add_pair(Precision::FP8_E5M2, Precision::FP8_E5M2);
        add_pair(Precision::INT16, Precision::INT16);

    } else if (layer_entry == "avg_pool2d") {

        // AVGPOOL: Has an accumulator, but NO weights.
        supports_squash_acc = true;
        has_weights = false;

        add_pair(Precision::FP16, std::nullopt);
        add_pair(Precision::INT8, std::nullopt);
        add_pair(Precision::BF16, std::nullopt);
        add_pair(Precision::FP8_E4M3, std::nullopt);
        add_pair(Precision::FP8_E5M2, std::nullopt);
        add_pair(Precision::INT16, std::nullopt);

    } else if (layer_entry == "fft2d" || layer_entry == "rfft2d") {

        // Strict FP32 only. No accumulators. No weights.
        // The default {std::nullopt, std::nullopt} pair handles this.
        supports_squash_acc = false;
        has_weights = false;

    } else if (layer_entry == "reduce_max" || layer_entry == "reduce_min") {

        // REDUCE MIN/MAX: No accumulator. No weights.
        supports_squash_acc = false;
        has_weights = false;

        add_pair(Precision::FP16, std::nullopt);
        add_pair(Precision::INT16, std::nullopt);
        add_pair(Precision::INT32, std::nullopt);
        add_pair(Precision::INT8, std::nullopt);
        add_pair(Precision::BF16, std::nullopt);

    } else if (layer_entry == "reduce_product") {

        // REDUCE PRODUCT: Limited support.
        supports_squash_acc = false;
        has_weights = false;

        add_pair(Precision::FP16, std::nullopt);
        add_pair(Precision::BF16, std::nullopt);

    } else if (layer_entry == "reduce_sum") {

        supports_squash_acc = false;
        has_weights = false;

        add_pair(Precision::FP16, std::nullopt);
        add_pair(Precision::INT32, std::nullopt);
        add_pair(Precision::BF16, std::nullopt);

    } else {
        // --------------------------------------------------------------------
        // ELEMENTWISE MATH OPERATIONS
        // --------------------------------------------------------------------
        // These ops have no accumulators. They CAN have weights (tosa.const)
        // if has_const_operand is true. If they do, the constant must be
        // quantised into the same mathematical domain/precision.
        supports_squash_acc = false;
        has_weights = has_const_operand;
        allow_per_channel_weights = false;

        auto add_elementwise_pair = [&](Precision p) {
            if (has_weights) add_pair(p, p);
            else             add_pair(p, std::nullopt);
        };

        if (layer_entry == "add" || layer_entry == "maximum" || layer_entry == "minimum" ||
            layer_entry == "sub" || layer_entry == "equal" ||
            layer_entry == "greater" || layer_entry == "greater_equal") {

            add_elementwise_pair(Precision::FP16);
            add_elementwise_pair(Precision::INT32);
            add_elementwise_pair(Precision::BF16);

        } else if (layer_entry == "mul") {

            add_elementwise_pair(Precision::FP16);
            add_elementwise_pair(Precision::INT16);
            add_elementwise_pair(Precision::INT32);
            add_elementwise_pair(Precision::INT8);
            add_elementwise_pair(Precision::BF16);

        } else if (layer_entry == "pow") {

            add_elementwise_pair(Precision::FP16);
            add_elementwise_pair(Precision::BF16);

        } else if (layer_entry == "arithmetic_right_shift") {

            add_elementwise_pair(Precision::INT16);
            add_elementwise_pair(Precision::INT32);
            add_elementwise_pair(Precision::INT8);

        }
    }

    // We now have a list of guaranteed-valid pairs for this specific operation.
    // We will expand these into full `Layer` configurations.

    std::vector<bool> squash_mutations = { false };
    if (supports_squash_acc) {
        squash_mutations.push_back(true);
    }

    for (const auto& pair : valid_pairs) {
        // Expand Activation Configs
        std::vector<LayerConf> act_configs;
        if (pair.act.has_value()) {
            act_configs = generate_act_configs(pair.act.value());
        } else {
            act_configs.emplace_back(SkipConfig{});
        }

        // Expand Weight Configs
        std::vector<LayerConf> weight_configs;
        if (pair.weight.has_value() && has_weights) {
            weight_configs = generate_weight_configs(pair.weight.value(), allow_per_channel_weights);
        } else {
            weight_configs.emplace_back(SkipConfig{});
        }

        // Cartesian Product of Valid Conf + Squash Status
        for (const auto& a_conf : act_configs) {
            for (const auto& w_conf : weight_configs) {
                for (const bool squash : squash_mutations) {

                    Layer new_layer;
                    new_layer.activation_policy = a_conf;
                    new_layer.weight_policy = w_conf;
                    new_layer.squash_acc = squash;

                    mutations.push_back(new_layer);
                }
            }
        }
    }

    // Cartesian product of all calibrations
    std::vector<Layer> final_mutations;

    for (const auto& layer : mutations) {
        bool has_quant = std::holds_alternative<QuantisationConfig>(layer.activation_policy) ||
                         std::holds_alternative<QuantisationConfig>(layer.weight_policy);

        // If it's a SkipConfig/SkipConfig layer, calibration doesn't apply
        if (!has_quant) {
            final_mutations.push_back(layer);
            continue;
        }

        std::vector<CalibrationMethod> calibrations = {
            CalibrationMethod::MinMax,
            CalibrationMethod::Entropy,
            CalibrationMethod::Percentile
        };

        if (allowed_calibrations.has_value() && !allowed_calibrations->empty()) {
            calibrations = *allowed_calibrations;
        }

        for (const auto& calib : calibrations) {
            Layer modified_layer = layer; // Copy the original layer

            if (std::holds_alternative<QuantisationConfig>(modified_layer.activation_policy)) {
                std::get<QuantisationConfig>(modified_layer.activation_policy).calibration = calib;
            }
            if (std::holds_alternative<QuantisationConfig>(modified_layer.weight_policy)) {
                std::get<QuantisationConfig>(modified_layer.weight_policy).calibration = calib;
            }

            final_mutations.push_back(modified_layer);
        }
    }

    return final_mutations;
    }
} // namespace conquer