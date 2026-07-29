#include "conquer/passes/analysis/model_proxy.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace conquer {
namespace {

[[nodiscard]] constexpr std::size_t saturating_mul(
    const std::size_t lhs,
    const std::size_t rhs) noexcept {
    if (lhs == 0U || rhs == 0U) {
        return 0U;
    }
    if (lhs > (std::numeric_limits<std::size_t>::max() / rhs)) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs * rhs;
}

[[nodiscard]] bool has_op_name(
    mlir::Operation *const op,
    const llvm::StringRef full_name) noexcept {
    return op != nullptr && op->getName().getStringRef() == full_name;
}

[[nodiscard]] bool is_tosa_const(mlir::Operation *const op) noexcept {
    return has_op_name(op, "tosa.const");
}

[[nodiscard]] std::optional<Precision> get_precision_from_type(mlir::Type type) {
    if (const auto tensor_type = mlir::dyn_cast<mlir::TensorType>(type)) {
        type = tensor_type.getElementType();
    }

    if (type.isF32()) {
        return Precision::FP32;
    }
    if (type.isF16()) {
        return Precision::FP16;
    }
    if (type.isBF16()) {
        return Precision::BF16;
    }
    if (type.isF8E4M3FN()) {
        return Precision::FP8_E4M3;
    }
    if (type.isF8E5M2()) {
        return Precision::FP8_E5M2;
    }

    if (const auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
        switch (integer_type.getWidth()) {
            case 4:
                return Precision::INT4;
            case 8:
                return Precision::INT8;
            case 16:
                return Precision::INT16;
            case 32:
                return Precision::INT32;
            case 48:
                return Precision::INT32; // TOSA accumulator-like fallback
            case 64:
                return Precision::INT32; // Conservative internal fallback
            default:
                break;
        }
    }

    return std::nullopt;
}

[[nodiscard]] int bit_width_of_precision(const Precision p) noexcept {
    switch (p) {
        case Precision::INT4:
            return 4;
        case Precision::INT8:
        case Precision::FP8_E4M3:
        case Precision::FP8_E5M2:
            return 8;
        case Precision::INT16:
        case Precision::FP16:
        case Precision::BF16:
            return 16;
        case Precision::INT32:
        case Precision::FP32:
        default:
            return 32;
    }
}

[[nodiscard]] bool is_small_integer_precision(const Precision p) noexcept {
    return p == Precision::INT4 || p == Precision::INT8 || p == Precision::INT16;
}

[[nodiscard]] std::size_t get_num_elements(const mlir::Type type) noexcept {
    const auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
    if (!shaped_type || !shaped_type.hasStaticShape()) {
        return 0U;
    }
    return static_cast<std::size_t>(shaped_type.getNumElements());
}

[[nodiscard]] std::vector<int64_t> get_static_shape(const mlir::Type type) {
    const auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
    if (!shaped_type || !shaped_type.hasStaticShape()) {
        return {};
    }

    return std::vector<int64_t>(
        shaped_type.getShape().begin(),
        shaped_type.getShape().end());
}

[[nodiscard]] std::size_t get_tensor_bytes(const mlir::Type type) noexcept {
    const auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
    if (!shaped_type || !shaped_type.hasStaticShape()) {
        return 0U;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(shaped_type.getNumElements());
    const std::size_t bit_width =
        static_cast<std::size_t>(shaped_type.getElementTypeBitWidth());

    return (saturating_mul(element_count, bit_width) + 7U) / 8U;
}

[[nodiscard]] std::optional<int64_t> get_positive_dim(
    const std::vector<int64_t> &shape,
    const std::size_t index) noexcept {
    if (index >= shape.size() || shape[index] <= 0) {
        return std::nullopt;
    }
    return shape[index];
}

[[nodiscard]] std::optional<int64_t> positive_product_to_i64(
    const std::initializer_list<int64_t> dims) noexcept {
    std::size_t product = 1U;
    for (const int64_t dim : dims) {
        if (dim <= 0) {
            return std::nullopt;
        }
        product = saturating_mul(product, static_cast<std::size_t>(dim));
    }

    constexpr std::size_t max_i64 =
        static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
    if (product > max_i64) {
        return std::numeric_limits<int64_t>::max();
    }

    return static_cast<int64_t>(product);
}

[[nodiscard]] std::optional<int64_t> get_optional_i64_attr(
    mlir::Operation *const op,
    const llvm::StringRef name) {
    if (const auto attr = op->getAttrOfType<mlir::IntegerAttr>(name)) {
        return attr.getInt();
    }
    return std::nullopt;
}

[[nodiscard]] bool get_bool_attr_or_default(
    mlir::Operation *const op,
    const llvm::StringRef name,
    const bool default_value = false) noexcept {
    if (const auto attr = op->getAttrOfType<mlir::BoolAttr>(name)) {
        return attr.getValue();
    }
    return default_value;
}

[[nodiscard]] std::vector<int64_t> get_i64_values_from_attr(
    const mlir::Attribute attr) {
    std::vector<int64_t> values;

    if (const auto dense_i64 = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
        const auto arr = dense_i64.asArrayRef();
        values.assign(arr.begin(), arr.end());
        return values;
    }

    if (const auto dense_ints = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
        values.reserve(static_cast<std::size_t>(dense_ints.getNumElements()));
        for (const int64_t value : dense_ints.getValues<int64_t>()) {
            values.push_back(value);
        }
        return values;
    }

    if (const auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
        values.reserve(array_attr.size());
        for (const mlir::Attribute element : array_attr) {
            const auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(element);
            if (!int_attr) {
                return {};
            }
            values.push_back(int_attr.getInt());
        }
        return values;
    }

    return {};
}

[[nodiscard]] std::vector<int64_t> get_i64_array_attr(
    mlir::Operation *const op,
    const llvm::StringRef name) {
    if (const mlir::Attribute attr = op->getAttr(name)) {
        return get_i64_values_from_attr(attr);
    }
    return {};
}

[[nodiscard]] std::vector<int64_t> get_constant_i64_values_from_value(
    const mlir::Value value) {
    mlir::Operation *const defining_op = value.getDefiningOp();
    if (!is_tosa_const(defining_op)) {
        return {};
    }

    mlir::Attribute attr = defining_op->getAttr("value");
    if (!attr) {
        attr = defining_op->getAttr("values");
    }
    if (!attr) {
        return {};
    }

    return get_i64_values_from_attr(attr);
}

[[nodiscard]] std::optional<float> get_float_attr(mlir::Operation *const op, const llvm::StringRef name) {
    if (const auto attr = op->getAttrOfType<mlir::FloatAttr>(name)) {
        return attr.getValueAsDouble();
    }
    return std::nullopt;
}

[[nodiscard]] bool all_ones(const std::vector<int64_t> &values) noexcept {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const int64_t value) { return value == 1; });
}

[[nodiscard]] bool has_broadcasting_operands(mlir::Operation *const op) {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(op->getNumOperands());

    for (const mlir::Value operand : op->getOperands()) {
        const auto shape = get_static_shape(operand.getType());
        if (!shape.empty() || get_num_elements(operand.getType()) == 1U) {
            shapes.push_back(shape);
        }
    }

    if (shapes.size() < 2U) {
        return false;
    }

    std::size_t max_rank = 0U;
    for (const auto &shape : shapes) {
        max_rank = std::max(max_rank, shape.size());
    }

    std::vector<int64_t> reference(max_rank, 1);
    {
        const auto &shape = shapes.front();
        const std::size_t offset = max_rank - shape.size();
        for (std::size_t i = 0; i < shape.size(); ++i) {
            reference[offset + i] = shape[i];
        }
    }

    for (std::size_t shape_index = 1U; shape_index < shapes.size(); ++shape_index) {
        std::vector<int64_t> normalized(max_rank, 1);
        const auto &shape = shapes[shape_index];
        const std::size_t offset = max_rank - shape.size();

        for (std::size_t i = 0; i < shape.size(); ++i) {
            normalized[offset + i] = shape[i];
        }

        if (normalized != reference) {
            return true;
        }
    }

    return false;
}

[[nodiscard]] AlignmentRemainders make_alignment_remainders(
    const std::optional<int64_t> dim) noexcept {
    if (!dim.has_value() || *dim <= 0) {
        return AlignmentRemainders{};
    }

    return AlignmentRemainders{
        .mod8 = (*dim % 8),
        .mod16 = (*dim % 16),
        .mod32 = (*dim % 32)
    };
}

[[nodiscard]] OpKind classify_kind(mlir::Operation *const op) {
    const llvm::StringRef name = op->getName().getStringRef();

    if (name == "tosa.matmul") return OpKind::MatMul;
    if (name == "tosa.conv2d") return OpKind::Conv2D;
    if (name == "tosa.conv3d") return OpKind::Conv3D;
    if (name == "tosa.depthwise_conv2d") return OpKind::DepthwiseConv2D;
    if (name == "tosa.transpose_conv2d") return OpKind::TransposeConv2D;

    if (name == "tosa.cast") return OpKind::Cast;
    if (name == "tosa.rescale") return OpKind::Rescale;

    if (name == "tosa.transpose") return OpKind::Transpose;
    if (name == "tosa.reshape") return OpKind::Reshape;
    if (name == "tosa.slice") return OpKind::Slice;
    if (name == "tosa.concat") return OpKind::Concat;
    if (name == "tosa.pad") return OpKind::Pad;
    if (name == "tosa.gather") return OpKind::Gather;
    if (name == "tosa.scatter") return OpKind::Scatter;
    if (name == "tosa.tile") return OpKind::Tile;
    if (name == "tosa.reverse") return OpKind::Reverse;

    if (name == "tosa.reduce_sum") return OpKind::ReduceSum;
    if (name == "tosa.reduce_product") return OpKind::ReduceProduct;
    if (name == "tosa.reduce_max") return OpKind::ReduceMax;
    if (name == "tosa.reduce_min") return OpKind::ReduceMin;

    if (name == "tosa.add") return OpKind::Add;
    if (name == "tosa.sub") return OpKind::Sub;
    if (name == "tosa.mul") return OpKind::Mul;
    if (name == "tosa.maximum") return OpKind::Maximum;
    if (name == "tosa.minimum") return OpKind::Minimum;
    if (name == "tosa.clamp") return OpKind::Clamp;
    if (name == "tosa.exp") return OpKind::Exp;
    if (name == "tosa.log") return OpKind::Log;
    if (name == "tosa.tanh") return OpKind::Tanh;
    if (name == "tosa.sigmoid") return OpKind::Sigmoid;
    if (name == "tosa.rsqrt") return OpKind::Rsqrt;
    if (name == "tosa.reciprocal") return OpKind::Reciprocal;
    if (name == "tosa.table") return OpKind::Table;

    llvm::StringRef stripped = name;
    if (stripped.consume_front("tosa.")) {
        return OpKind::GenericElementwise;
    }

    return OpKind::Unknown;
}

[[nodiscard]] OpCategory classify_category(const OpKind kind) noexcept {
    switch (kind) {
        case OpKind::MatMul:
        case OpKind::Conv2D:
        case OpKind::Conv3D:
        case OpKind::DepthwiseConv2D:
        case OpKind::TransposeConv2D:
            return OpCategory::Compute;

        case OpKind::Cast:
        case OpKind::Rescale:
            return OpCategory::Cast;

        case OpKind::Transpose:
        case OpKind::Reshape:
        case OpKind::Slice:
        case OpKind::Concat:
        case OpKind::Pad:
        case OpKind::Gather:
        case OpKind::Scatter:
        case OpKind::Tile:
        case OpKind::Reverse:
            return OpCategory::MemoryMovement;

        case OpKind::ReduceSum:
        case OpKind::ReduceProduct:
        case OpKind::ReduceMax:
        case OpKind::ReduceMin:
        case OpKind::Add:
        case OpKind::Sub:
        case OpKind::Mul:
        case OpKind::Maximum:
        case OpKind::Minimum:
        case OpKind::Clamp:
        case OpKind::Exp:
        case OpKind::Log:
        case OpKind::Tanh:
        case OpKind::Sigmoid:
        case OpKind::Rsqrt:
        case OpKind::Reciprocal:
        case OpKind::Table:
        case OpKind::GenericElementwise:
        case OpKind::Unknown:
        default:
            return OpCategory::Elementwise;
    }
}

[[nodiscard]] std::size_t calculate_macs(
    mlir::Operation *const op,
    const OpKind kind) {
    const auto mul_dims =
        [](const std::initializer_list<std::size_t> dims) noexcept -> std::size_t {
            std::size_t result = 1U;
            for (const std::size_t dim : dims) {
                result = saturating_mul(result, dim);
            }
            return result;
        };

    if (op->getNumResults() == 0U) {
        return 0U;
    }

    if (kind == OpKind::MatMul && op->getNumOperands() >= 2U) {
        const auto a_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(0).getType()).getShape();
        const auto b_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(1).getType()).getShape();

        if (a_shape.size() == 3U && b_shape.size() == 3U &&
            a_shape[0] > 0 && a_shape[1] > 0 && a_shape[2] > 0 && b_shape[2] > 0) {
            return mul_dims({
                static_cast<std::size_t>(a_shape[0]),
                static_cast<std::size_t>(a_shape[1]),
                static_cast<std::size_t>(a_shape[2]),
                static_cast<std::size_t>(b_shape[2]),
            });
        }
        return 0U;
    }

    if (kind == OpKind::Conv2D && op->getNumOperands() >= 2U) {
        const auto wt_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(1).getType()).getShape();
        const auto out_shape =
            mlir::cast<mlir::ShapedType>(op->getResult(0).getType()).getShape();

        if (wt_shape.size() == 4U && out_shape.size() == 4U &&
            wt_shape[1] > 0 && wt_shape[2] > 0 && wt_shape[3] > 0 &&
            out_shape[0] > 0 && out_shape[1] > 0 && out_shape[2] > 0 && out_shape[3] > 0) {
            return mul_dims({
                static_cast<std::size_t>(out_shape[0]),
                static_cast<std::size_t>(out_shape[1]),
                static_cast<std::size_t>(out_shape[2]),
                static_cast<std::size_t>(out_shape[3]),
                static_cast<std::size_t>(wt_shape[1]),
                static_cast<std::size_t>(wt_shape[2]),
                static_cast<std::size_t>(wt_shape[3]),
            });
        }
        return 0U;
    }

    if (kind == OpKind::Conv3D && op->getNumOperands() >= 2U) {
        const auto wt_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(1).getType()).getShape();
        const auto out_shape =
            mlir::cast<mlir::ShapedType>(op->getResult(0).getType()).getShape();

        if (wt_shape.size() == 5U && out_shape.size() == 5U &&
            wt_shape[1] > 0 && wt_shape[2] > 0 && wt_shape[3] > 0 && wt_shape[4] > 0 &&
            out_shape[0] > 0 && out_shape[1] > 0 && out_shape[2] > 0 &&
            out_shape[3] > 0 && out_shape[4] > 0) {
            return mul_dims({
                static_cast<std::size_t>(out_shape[0]),
                static_cast<std::size_t>(out_shape[1]),
                static_cast<std::size_t>(out_shape[2]),
                static_cast<std::size_t>(out_shape[3]),
                static_cast<std::size_t>(out_shape[4]),
                static_cast<std::size_t>(wt_shape[1]),
                static_cast<std::size_t>(wt_shape[2]),
                static_cast<std::size_t>(wt_shape[3]),
                static_cast<std::size_t>(wt_shape[4]),
            });
        }
        return 0U;
    }

    if (kind == OpKind::DepthwiseConv2D && op->getNumOperands() >= 2U) {
        const auto wt_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(1).getType()).getShape();
        const auto out_shape =
            mlir::cast<mlir::ShapedType>(op->getResult(0).getType()).getShape();

        if (wt_shape.size() == 4U && out_shape.size() == 4U &&
            wt_shape[0] > 0 && wt_shape[1] > 0 && wt_shape[2] > 0 && wt_shape[3] > 0 &&
            out_shape[0] > 0 && out_shape[1] > 0 && out_shape[2] > 0) {
            return mul_dims({
                static_cast<std::size_t>(out_shape[0]),
                static_cast<std::size_t>(out_shape[1]),
                static_cast<std::size_t>(out_shape[2]),
                static_cast<std::size_t>(wt_shape[0]),
                static_cast<std::size_t>(wt_shape[1]),
                static_cast<std::size_t>(wt_shape[2]),
                static_cast<std::size_t>(wt_shape[3]),
            });
        }
        return 0U;
    }

    if (kind == OpKind::TransposeConv2D && op->getNumOperands() >= 2U) {
        const auto wt_shape =
            mlir::cast<mlir::ShapedType>(op->getOperand(1).getType()).getShape();
        const auto out_shape =
            mlir::cast<mlir::ShapedType>(op->getResult(0).getType()).getShape();

        if (wt_shape.size() == 4U && out_shape.size() == 4U &&
            wt_shape[1] > 0 && wt_shape[2] > 0 && wt_shape[3] > 0 &&
            out_shape[0] > 0 && out_shape[1] > 0 && out_shape[2] > 0 && out_shape[3] > 0) {
            return mul_dims({
                static_cast<std::size_t>(out_shape[0]),
                static_cast<std::size_t>(out_shape[1]),
                static_cast<std::size_t>(out_shape[2]),
                static_cast<std::size_t>(out_shape[3]),
                static_cast<std::size_t>(wt_shape[1]),
                static_cast<std::size_t>(wt_shape[2]),
                static_cast<std::size_t>(wt_shape[3]),
            });
        }
        return 0U;
    }

    return 0U;
}

[[nodiscard]] bool should_skip_op(mlir::Operation *const op) {
    return mlir::isa<
        mlir::ModuleOp,
        mlir::func::FuncOp,
        mlir::func::ReturnOp>(op) || is_tosa_const(op);
}

template <typename T>
void append_unique(std::vector<T> &values, const T &value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

[[nodiscard]] Precision choose_representative_output_precision(
    mlir::Operation *const op) {
    std::optional<Precision> best;

    for (const mlir::Value result : op->getResults()) {
        const auto precision = get_precision_from_type(result.getType());
        if (!precision.has_value()) {
            continue;
        }
        if (!best.has_value() ||
            bit_width_of_precision(*precision) > bit_width_of_precision(*best)) {
            best = precision;
        }
    }

    return best.value_or(Precision::FP32);
}

[[nodiscard]] std::optional<Precision> infer_accumulator_type(
    mlir::Operation *const op,
    const OpCategory category,
    const Precision output_type) {
    if (category != OpCategory::Compute) {
        return std::nullopt;
    }

    std::vector<Precision> mul_input_precisions;
    for (unsigned index = 0U; index < std::min<unsigned>(2U, op->getNumOperands()); ++index) {
        const auto precision = get_precision_from_type(op->getOperand(index).getType());
        if (precision.has_value()) {
            mul_input_precisions.push_back(*precision);
        }
    }

    if (mul_input_precisions.size() >= 2U &&
        std::all_of(
            mul_input_precisions.begin(),
            mul_input_precisions.end(),
            [](const Precision p) { return is_small_integer_precision(p); })) {
        return Precision::INT32;
    }

    return output_type;
}

void enrich_node_static_features(mlir::Operation *const op, DAGNode &node) {
    if (op->getNumResults() > 0U) {
        node.output_shape = get_static_shape(op->getResult(0).getType());
        node.output_elements = get_num_elements(op->getResult(0).getType());
    }

    const bool can_broadcast =
        node.kind == OpKind::Add ||
        node.kind == OpKind::Sub ||
        node.kind == OpKind::Mul ||
        node.kind == OpKind::Maximum ||
        node.kind == OpKind::Minimum ||
        node.kind == OpKind::Clamp ||
        node.kind == OpKind::GenericElementwise;

    node.has_broadcast = can_broadcast && has_broadcasting_operands(op);

    node.axis = get_optional_i64_attr(op, "axis");
    node.slice_start = get_i64_array_attr(op, "start");
    node.slice_size = get_i64_array_attr(op, "size");
    node.strides = get_i64_array_attr(op, "stride");
    node.dilations = get_i64_array_attr(op, "dilation");
    node.pads = get_i64_array_attr(op, "pad");

    if (node.kind == OpKind::Transpose && op->getNumOperands() > 1U) {
        node.permutation = get_constant_i64_values_from_value(op->getOperand(1));
    }

    if (node.kind == OpKind::Pad && node.pads.empty() && op->getNumOperands() > 1U) {
        node.pads = get_constant_i64_values_from_value(op->getOperand(1));
    }

    if ((node.kind == OpKind::ReduceSum ||
         node.kind == OpKind::ReduceProduct ||
         node.kind == OpKind::ReduceMax ||
         node.kind == OpKind::ReduceMin) &&
        node.axis.has_value()) {
        node.reduce_axes.push_back(*node.axis);
    }

    node.input_zero_point = get_optional_i64_attr(op, "input_zp");
    node.weight_zero_point = get_optional_i64_attr(op, "weight_zp");
    node.output_zero_point = get_optional_i64_attr(op, "output_zp");

    node.activation_sensitivity = get_float_attr(op, "sensitivity_value.entropy.activation");

    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        const std::string attr_name = "sensitivity_value.entropy.weight_" + std::to_string(i);
        if (const auto weight_sens = get_float_attr(op, attr_name)) {
            node.weight_sensitivity.push_back(*weight_sens);
        }
    }

    node.is_per_channel_quantized = get_bool_attr_or_default(op, "per_channel", false);
    node.rescale_scale32 = get_bool_attr_or_default(op, "scale32", false);
    node.rescale_double_round = get_bool_attr_or_default(op, "double_round", false);

    const auto has_non_zero = [](const std::optional<int64_t> &value) noexcept {
        return value.has_value() && *value != 0;
    };

    node.has_non_zero_zp =
        has_non_zero(node.input_zero_point) ||
        has_non_zero(node.weight_zero_point) ||
        has_non_zero(node.output_zero_point);

    node.is_view_like = (node.kind == OpKind::Reshape);
    node.is_layout_changing =
        node.kind == OpKind::Transpose ||
        node.kind == OpKind::Reverse ||
        node.kind == OpKind::Gather ||
        node.kind == OpKind::Scatter ||
        node.kind == OpKind::Tile;

    const std::vector<int64_t> in0 =
        op->getNumOperands() > 0U ? get_static_shape(op->getOperand(0).getType()) : std::vector<int64_t>{};
    const std::vector<int64_t> in1 =
        op->getNumOperands() > 1U ? get_static_shape(op->getOperand(1).getType()) : std::vector<int64_t>{};
    const std::vector<int64_t> out = node.output_shape;

    std::optional<int64_t> c_in;
    std::optional<int64_t> c_out;

    switch (node.kind) {
        case OpKind::MatMul:
            if (in0.size() == 3U && in1.size() == 3U) {
                node.batch = get_positive_dim(in0, 0U);
                node.m = get_positive_dim(in0, 1U);
                node.k = get_positive_dim(in0, 2U);
                node.n = get_positive_dim(in1, 2U);
                c_in = node.k;
                c_out = node.n;
            }
            break;

        case OpKind::Conv2D:
            if (in0.size() == 4U && in1.size() == 4U && out.size() == 4U) {
                node.batch = get_positive_dim(in0, 0U);
                node.kernel_shape = {
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1)
                };
                node.m = positive_product_to_i64({
                    get_positive_dim(out, 0U).value_or(-1),
                    get_positive_dim(out, 1U).value_or(-1),
                    get_positive_dim(out, 2U).value_or(-1)
                });
                node.n = get_positive_dim(out, 3U);
                node.k = positive_product_to_i64({
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1),
                    get_positive_dim(in1, 3U).value_or(-1)
                });
                c_in = get_positive_dim(in0, 3U);
                c_out = get_positive_dim(out, 3U);

                node.is_pointwise_1x1 =
                    node.kernel_shape.size() == 2U &&
                    node.kernel_shape[0] == 1 &&
                    node.kernel_shape[1] == 1 &&
                    (node.strides.empty() || all_ones(node.strides)) &&
                    (node.dilations.empty() || all_ones(node.dilations));
            }
            break;

        case OpKind::Conv3D:
            if (in0.size() == 5U && in1.size() == 5U && out.size() == 5U) {
                node.batch = get_positive_dim(in0, 0U);
                node.kernel_shape = {
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1),
                    get_positive_dim(in1, 3U).value_or(-1)
                };
                node.m = positive_product_to_i64({
                    get_positive_dim(out, 0U).value_or(-1),
                    get_positive_dim(out, 1U).value_or(-1),
                    get_positive_dim(out, 2U).value_or(-1),
                    get_positive_dim(out, 3U).value_or(-1)
                });
                node.n = get_positive_dim(out, 4U);
                node.k = positive_product_to_i64({
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1),
                    get_positive_dim(in1, 3U).value_or(-1),
                    get_positive_dim(in1, 4U).value_or(-1)
                });
                c_in = get_positive_dim(in0, 4U);
                c_out = get_positive_dim(out, 4U);
            }
            break;

        case OpKind::DepthwiseConv2D:
            if (in0.size() == 4U && in1.size() == 4U && out.size() == 4U) {
                node.is_depthwise = true;
                node.batch = get_positive_dim(in0, 0U);
                node.kernel_shape = {
                    get_positive_dim(in1, 0U).value_or(-1),
                    get_positive_dim(in1, 1U).value_or(-1)
                };
                node.m = positive_product_to_i64({
                    get_positive_dim(out, 0U).value_or(-1),
                    get_positive_dim(out, 1U).value_or(-1),
                    get_positive_dim(out, 2U).value_or(-1)
                });
                node.n = get_positive_dim(out, 3U);
                node.k = positive_product_to_i64({
                    get_positive_dim(in1, 0U).value_or(-1),
                    get_positive_dim(in1, 1U).value_or(-1)
                });
                c_in = get_positive_dim(in0, 3U);
                c_out = get_positive_dim(out, 3U);

                node.is_pointwise_1x1 =
                    node.kernel_shape.size() == 2U &&
                    node.kernel_shape[0] == 1 &&
                    node.kernel_shape[1] == 1 &&
                    (node.strides.empty() || all_ones(node.strides)) &&
                    (node.dilations.empty() || all_ones(node.dilations));
            }
            break;

        case OpKind::TransposeConv2D:
            if (in0.size() == 4U && in1.size() == 4U && out.size() == 4U) {
                node.batch = get_positive_dim(in0, 0U);
                node.kernel_shape = {
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1)
                };
                node.m = positive_product_to_i64({
                    get_positive_dim(out, 0U).value_or(-1),
                    get_positive_dim(out, 1U).value_or(-1),
                    get_positive_dim(out, 2U).value_or(-1)
                });
                node.n = get_positive_dim(out, 3U);
                node.k = positive_product_to_i64({
                    get_positive_dim(in1, 1U).value_or(-1),
                    get_positive_dim(in1, 2U).value_or(-1),
                    get_positive_dim(in1, 3U).value_or(-1)
                });
                c_in = get_positive_dim(in0, 3U);
                c_out = get_positive_dim(out, 3U);

                node.is_pointwise_1x1 =
                    node.kernel_shape.size() == 2U &&
                    node.kernel_shape[0] == 1 &&
                    node.kernel_shape[1] == 1 &&
                    (node.strides.empty() || all_ones(node.strides)) &&
                    (node.dilations.empty() || all_ones(node.dilations));
            }
            break;

        default:
            break;
    }

    node.m_remainder = make_alignment_remainders(node.m);
    node.n_remainder = make_alignment_remainders(node.n);
    node.k_remainder = make_alignment_remainders(node.k);
    node.c_in_remainder = make_alignment_remainders(c_in);
    node.c_out_remainder = make_alignment_remainders(c_out);
}

[[nodiscard]] std::size_t estimate_compute_ops(
    mlir::Operation *const op,
    const DAGNode &node) noexcept {
    const std::size_t input0_elements =
        op->getNumOperands() > 0U ? get_num_elements(op->getOperand(0).getType()) : 0U;

    switch (node.kind) {
        case OpKind::MatMul:
        case OpKind::Conv2D:
        case OpKind::Conv3D:
        case OpKind::DepthwiseConv2D:
        case OpKind::TransposeConv2D:
            return node.mac_count;

        case OpKind::Cast:
            return node.output_elements;

        case OpKind::Rescale:
            return saturating_mul(node.output_elements, 4U);

        case OpKind::ReduceSum:
        case OpKind::ReduceProduct:
        case OpKind::ReduceMax:
        case OpKind::ReduceMin:
            return std::max(input0_elements, node.output_elements);

        case OpKind::Exp:
        case OpKind::Log:
        case OpKind::Tanh:
        case OpKind::Sigmoid:
        case OpKind::Rsqrt:
        case OpKind::Reciprocal:
        case OpKind::Table:
            return saturating_mul(node.output_elements, 8U);

        case OpKind::Add:
        case OpKind::Sub:
        case OpKind::Mul:
        case OpKind::Maximum:
        case OpKind::Minimum:
        case OpKind::Clamp:
        case OpKind::GenericElementwise:
            return node.output_elements;

        case OpKind::Unknown:
        default:
            return node.category == OpCategory::Elementwise ? node.output_elements : 0U;
    }
}

} // namespace

QuantizedDAG build_quantized_dag(mlir::ModuleOp module) {
    QuantizedDAG dag{};

    std::vector<mlir::Operation *> ordered_ops;
    ordered_ops.reserve(256U);

    module.walk([&](mlir::Operation *const op) {
        if (!should_skip_op(op)) {
            ordered_ops.push_back(op);
        }
    });

    llvm::DenseMap<mlir::Operation *, int> op_to_node_index;
    llvm::DenseMap<mlir::Value, int> value_to_node_index;
    llvm::DenseMap<mlir::Value, std::size_t> value_bytes;
    llvm::DenseMap<mlir::Value, std::size_t> remaining_uses;
    llvm::DenseSet<mlir::Operation *> counted_constant_ops;

    // Pre-register block arguments so activation liveness can be estimated.
    module.walk([&](mlir::func::FuncOp func) {
        for (const mlir::BlockArgument argument : func.getArguments()) {
            const std::size_t bytes = get_tensor_bytes(argument.getType());
            if (bytes > 0U) {
                value_bytes[argument] = bytes;
            }
        }
    });

    // Pre-register result sizes for all kept ops.
    for (mlir::Operation *const op : ordered_ops) {
        for (const mlir::Value result : op->getResults()) {
            const std::size_t bytes = get_tensor_bytes(result.getType());
            if (bytes > 0U) {
                value_bytes[result] = bytes;
            }
        }
    }

    // Count activation uses.
    for (mlir::Operation *const op : ordered_ops) {
        for (const mlir::Value operand : op->getOperands()) {
            mlir::Operation *const defining_op = operand.getDefiningOp();
            const bool is_constant = is_tosa_const(defining_op);
            if (is_constant) {
                continue;
            }

            if (value_bytes.find(operand) != value_bytes.end()) {
                ++remaining_uses[operand];
            }
        }
    }

    for (mlir::Operation *const op : ordered_ops) {
        DAGNode node{};
        node.op_name = op->getName().getStringRef().str();
        node.layer_id =
            op->getAttrOfType<mlir::StringAttr>("conquer.name")
                ? op->getAttrOfType<mlir::StringAttr>("conquer.name").getValue().str()
                : (node.op_name + "_" + std::to_string(dag.nodes.size()));

        node.kind = classify_kind(op);
        node.category = classify_category(node.kind);
        node.mac_count = calculate_macs(op, node.kind);
        node.output_type = choose_representative_output_precision(op);
        node.accumulator_type = infer_accumulator_type(op, node.category, node.output_type);

        for (const mlir::Value operand : op->getOperands()) {
            OperandStats stats{};
            stats.bytes = get_tensor_bytes(operand.getType());
            stats.dtype =
                get_precision_from_type(operand.getType()).value_or(Precision::FP32);
            stats.shape = get_static_shape(operand.getType());
            stats.num_elements = get_num_elements(operand.getType());

            if (mlir::Operation *const defining_op = operand.getDefiningOp()) {
                if (is_tosa_const(defining_op)) {
                    stats.is_constant = true;
                    if (counted_constant_ops.insert(defining_op).second) {
                        dag.total_model_parameters_bytes += stats.bytes;
                    }
                } else {
                    stats.is_constant = false;
                    if (const auto it = value_to_node_index.find(operand);
                        it != value_to_node_index.end()) {
                        append_unique(node.parent_indices, it->second);
                    } else if (const auto op_it = op_to_node_index.find(defining_op);
                               op_it != op_to_node_index.end()) {
                        append_unique(node.parent_indices, op_it->second);
                    }
                }
            } else {
                stats.is_constant = false;
            }

            node.inputs.push_back(std::move(stats));
        }

        node.total_output_bytes = 0U;
        for (const mlir::Value result : op->getResults()) {
            node.total_output_bytes += get_tensor_bytes(result.getType());
        }

        enrich_node_static_features(op, node);
        node.estimated_compute_ops = estimate_compute_ops(op, node);

        const int node_index = static_cast<int>(dag.nodes.size());
        op_to_node_index[op] = node_index;
        for (const mlir::Value result : op->getResults()) {
            value_to_node_index[result] = node_index;
        }

        dag.nodes.push_back(std::move(node));
    }

    // Fill children.
    for (int child_index = 0; child_index < static_cast<int>(dag.nodes.size()); ++child_index) {
        for (const int parent_index : dag.nodes[static_cast<std::size_t>(child_index)].parent_indices) {
            append_unique(
                dag.nodes[static_cast<std::size_t>(parent_index)].child_indices,
                child_index);
        }
    }

    // Approximate activation peak using SSA liveness.
    std::size_t current_live_bytes = 0U;
    for (const auto &[value, uses] : remaining_uses) {
        if (uses > 0U) {
            const auto it = value_bytes.find(value);
            if (it != value_bytes.end()) {
                current_live_bytes += it->second;
            }
        }
    }

    std::size_t peak_live_bytes = current_live_bytes;

    for (mlir::Operation *const op : ordered_ops) {
        // Materialize outputs first.
        for (const mlir::Value result : op->getResults()) {
            const auto it = value_bytes.find(result);
            if (it != value_bytes.end()) {
                current_live_bytes += it->second;
            }
        }

        peak_live_bytes = std::max(peak_live_bytes, current_live_bytes);

        // Retire inputs whose final use has just happened.
        for (const mlir::Value operand : op->getOperands()) {
            const auto uses_it = remaining_uses.find(operand);
            if (uses_it == remaining_uses.end() || uses_it->second == 0U) {
                continue;
            }

            --uses_it->second;
            if (uses_it->second == 0U) {
                const auto bytes_it = value_bytes.find(operand);
                if (bytes_it != value_bytes.end()) {
                    current_live_bytes =
                        (current_live_bytes >= bytes_it->second)
                            ? (current_live_bytes - bytes_it->second)
                            : 0U;
                }
            }
        }
    }

    dag.peak_activation_memory_bytes = peak_live_bytes;
    return dag;
}

} // namespace conquer
