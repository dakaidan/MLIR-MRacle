#pragma once

#include "conquer/quantisation/policy.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace conquer {
    enum class OpCategory {
        Compute,         // MatMuls, Convs
        MemoryMovement,  // Transpose, Reshape, Concat
        Elementwise,     // Add, Mul, Exp, Reduce, etc.
        Cast             // Type conversion / rescale overhead
    };

    enum class OpKind {
        Unknown,

        MatMul,
        Conv2D,
        Conv3D,
        DepthwiseConv2D,
        TransposeConv2D,

        Cast,
        Rescale,

        Transpose,
        Reshape,
        Slice,
        Concat,
        Pad,
        Gather,
        Scatter,
        Tile,
        Reverse,

        ReduceSum,
        ReduceProduct,
        ReduceMax,
        ReduceMin,

        Add,
        Sub,
        Mul,
        Maximum,
        Minimum,
        Clamp,
        Exp,
        Log,
        Tanh,
        Sigmoid,
        Rsqrt,
        Reciprocal,
        Table,

        GenericElementwise
    };

    struct AlignmentRemainders {
        int64_t mod8 = -1;
        int64_t mod16 = -1;
        int64_t mod32 = -1;
    };

    struct OperandStats {
        Precision dtype = Precision::FP32;
        std::size_t bytes = 0U;
        bool is_constant = false;

        std::vector<int64_t> shape;
        std::size_t num_elements = 0U;
    };

    struct DAGNode {
        std::string op_name;
        std::string layer_id;
        OpCategory category = OpCategory::Elementwise;
        OpKind kind = OpKind::Unknown;

        std::vector<OperandStats> inputs;

        std::optional<Precision> accumulator_type;
        Precision output_type = Precision::FP32;
        std::vector<int64_t> output_shape;
        std::size_t total_output_bytes = 0U;
        std::size_t output_elements = 0U;

        std::size_t mac_count = 0U;
        std::size_t estimated_compute_ops = 0U;

        std::optional<int64_t> batch;
        std::optional<int64_t> m;
        std::optional<int64_t> n;
        std::optional<int64_t> k;

        std::vector<int64_t> kernel_shape;
        std::vector<int64_t> strides;
        std::vector<int64_t> dilations;
        std::vector<int64_t> pads;

        std::optional<int64_t> axis;
        std::vector<int64_t> permutation;
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> slice_start;
        std::vector<int64_t> slice_size;

        std::optional<int64_t> input_zero_point;
        std::optional<int64_t> weight_zero_point;
        std::optional<int64_t> output_zero_point;

        std::vector<float> weight_sensitivity;
        std::optional<float> activation_sensitivity;

        bool has_broadcast = false;
        bool is_view_like = false;
        bool is_layout_changing = false;
        bool is_per_channel_quantized = false;
        bool has_non_zero_zp = false;
        bool rescale_scale32 = false;
        bool rescale_double_round = false;
        bool is_depthwise = false;
        bool is_pointwise_1x1 = false;

        AlignmentRemainders m_remainder;
        AlignmentRemainders n_remainder;
        AlignmentRemainders k_remainder;
        AlignmentRemainders c_in_remainder;
        AlignmentRemainders c_out_remainder;

        std::vector<int> parent_indices;
        std::vector<int> child_indices;
    };

    struct QuantizedDAG {
        std::vector<DAGNode> nodes;
        std::size_t total_model_parameters_bytes = 0U;
        std::size_t peak_activation_memory_bytes = 0U;
    };

    [[nodiscard]] QuantizedDAG build_quantized_dag(mlir::ModuleOp module);

    using json = nlohmann::json;

    NLOHMANN_JSON_SERIALIZE_ENUM(OpCategory, {
        {OpCategory::Compute, "Compute"},
        {OpCategory::MemoryMovement, "MemoryMovement"},
        {OpCategory::Elementwise, "Elementwise"},
        {OpCategory::Cast, "Cast"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(OpKind, {
        {OpKind::Unknown, "unknown"},

        {OpKind::MatMul, "matmul"},
        {OpKind::Conv2D, "conv2d"},
        {OpKind::Conv3D, "conv3d"},
        {OpKind::DepthwiseConv2D, "depthwise_conv2d"},
        {OpKind::TransposeConv2D, "transpose_conv2d"},

        {OpKind::Cast, "cast"},
        {OpKind::Rescale, "rescale"},

        {OpKind::Transpose, "transpose"},
        {OpKind::Reshape, "reshape"},
        {OpKind::Slice, "slice"},
        {OpKind::Concat, "concat"},
        {OpKind::Pad, "pad"},
        {OpKind::Gather, "gather"},
        {OpKind::Scatter, "scatter"},
        {OpKind::Tile, "tile"},
        {OpKind::Reverse, "reverse"},

        {OpKind::ReduceSum, "reduce_sum"},
        {OpKind::ReduceProduct, "reduce_product"},
        {OpKind::ReduceMax, "reduce_max"},
        {OpKind::ReduceMin, "reduce_min"},

        {OpKind::Add, "add"},
        {OpKind::Sub, "sub"},
        {OpKind::Mul, "mul"},
        {OpKind::Maximum, "maximum"},
        {OpKind::Minimum, "minimum"},
        {OpKind::Clamp, "clamp"},
        {OpKind::Exp, "exp"},
        {OpKind::Log, "log"},
        {OpKind::Tanh, "tanh"},
        {OpKind::Sigmoid, "sigmoid"},
        {OpKind::Rsqrt, "rsqrt"},
        {OpKind::Reciprocal, "reciprocal"},
        {OpKind::Table, "table"},

        {OpKind::GenericElementwise, "generic_elementwise"},
    })

    inline void to_json(json& j, const AlignmentRemainders& r) {
        j = json{
                {"mod8", r.mod8},
                {"mod16", r.mod16},
                {"mod32", r.mod32}
        };
    }

    inline void to_json(json& j, const OperandStats& s) {
        j = json{
                {"dtype", to_string(s.dtype)},
                {"bytes", s.bytes},
                {"is_constant", s.is_constant},
                {"shape", s.shape},
                {"num_elements", s.num_elements}
        };
    }

    inline void to_json(json& j, const DAGNode& n) {
        j = json{
                {"op_name", n.op_name},
                {"layer_id", n.layer_id},
                {"category", n.category},
                {"kind", n.kind},
                {"inputs", n.inputs},
                {"output_type", to_string(n.output_type)},
                {"output_shape", n.output_shape},
                {"total_output_bytes", n.total_output_bytes},
                {"output_elements", n.output_elements},
                {"mac_count", n.mac_count},
                {"estimated_compute_ops", n.estimated_compute_ops},

                {"kernel_shape", n.kernel_shape},
                {"strides", n.strides},
                {"dilations", n.dilations},
                {"pads", n.pads},

                {"permutation", n.permutation},
                {"reduce_axes", n.reduce_axes},
                {"slice_start", n.slice_start},
                {"slice_size", n.slice_size},

                {"has_broadcast", n.has_broadcast},
                {"is_view_like", n.is_view_like},
                {"is_layout_changing", n.is_layout_changing},
                {"is_per_channel_quantized", n.is_per_channel_quantized},
                {"has_non_zero_zp", n.has_non_zero_zp},
                {"rescale_scale32", n.rescale_scale32},
                {"rescale_double_round", n.rescale_double_round},
                {"is_depthwise", n.is_depthwise},
                {"is_pointwise_1x1", n.is_pointwise_1x1},

                {"m_remainder", n.m_remainder},
                {"n_remainder", n.n_remainder},
                {"k_remainder", n.k_remainder},
                {"c_in_remainder", n.c_in_remainder},
                {"c_out_remainder", n.c_out_remainder},

                {"parent_indices", n.parent_indices},
                {"child_indices", n.child_indices}
        };

        if (n.accumulator_type) {
            j["accumulator_type"] = to_string(*n.accumulator_type);
        } else {
            j["accumulator_type"] = nullptr;
        }

        if (n.batch) {
            j["batch"] = *n.batch;
        } else {
            j["batch"] = nullptr;
        }

        if (n.m) {
            j["m"] = *n.m;
        } else {
            j["m"] = nullptr;
        }

        if (n.n) {
            j["n"] = *n.n;
        } else {
            j["n"] = nullptr;
        }

        if (n.k) {
            j["k"] = *n.k;
        } else {
            j["k"] = nullptr;
        }

        if (n.axis) {
            j["axis"] = *n.axis;
        } else {
            j["axis"] = nullptr;
        }

        if (n.input_zero_point) {
            j["input_zero_point"] = *n.input_zero_point;
        } else {
            j["input_zero_point"] = nullptr;
        }

        if (n.weight_zero_point) {
            j["weight_zero_point"] = *n.weight_zero_point;
        } else {
            j["weight_zero_point"] = nullptr;
        }

        if (n.output_zero_point) {
            j["output_zero_point"] = *n.output_zero_point;
        } else {
            j["output_zero_point"] = nullptr;
        }

        j["weight_sensitivity"] = n.weight_sensitivity;

        if (n.activation_sensitivity) {
            j["activation_sensitivity"] = *n.activation_sensitivity;
        } else {
            j["activation_sensitivity"] = nullptr;
        }
    }

    inline void to_json(json& j, const QuantizedDAG& d) {
        j = json{
                {"schema_version", 2},
                {"nodes", d.nodes},
                {"total_model_parameters_bytes", d.total_model_parameters_bytes},
                {"peak_activation_memory_bytes", d.peak_activation_memory_bytes}
        };
    }
} // namespace conquer
