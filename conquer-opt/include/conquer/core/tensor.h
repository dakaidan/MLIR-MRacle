#pragma once

#include "conquer/core/types.h"

#include <mlir/Pass/PassManager.h>

#include <llvm/Support/Format.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace conquer {
struct TensorView {
    void *data;
    std::vector<std::int64_t> shape;
    DataType dtype;
    // explicit stride information may be useful for the future
    // std::vector<int64_t> strides;

    // Helper to get total element count
    [[nodiscard]] std::size_t getNumElements() const {
        if (shape.empty())
            return 0;
        std::size_t count = 1;
        for (const auto s : shape)
            count *= s;
        return count;
    }
};

struct TensorAllocation {
    std::vector<char> buffer;
    std::vector<std::int64_t> shape;
    DataType dtype;

    TensorView getView() { return {buffer.data(), shape, dtype}; }

    [[nodiscard]] TensorView getView() const { return {const_cast<char *>(buffer.data()), shape, dtype}; }
};

inline std::ostream &operator<<(std::ostream &os, const TensorView &view) {
    os << "TensorView(Shape: [";
    for (std::size_t i = 0; i < view.shape.size(); ++i) {
        os << view.shape[i] << (i < view.shape.size() - 1 ? ", " : "");
    }
    os << "], DType: " << view.dtype << ", Size: " << view.getNumElements() << " elements)";

    const std::size_t total_elements = view.getNumElements();

    if (view.data != nullptr && total_elements > 0) {
        os << "\n\t";
        try {
            const std::size_t byte_stride = get_size_bytes(view.dtype);
            if (byte_stride == 0)
                throw std::invalid_argument("Unknown byte size");

            const std::size_t display_count = std::min(total_elements, static_cast<std::size_t>(16));

            const std::vector head_buffer(static_cast<const char *>(view.data),
                                          static_cast<const char *>(view.data) + (display_count * byte_stride));

            auto buffer_variant = unpack_buffer(head_buffer, view.dtype);

            std::visit(
                [&os, total_elements]<typename T0>(const T0 &vec) {
                    using T = std::decay_t<T0>;

                    if constexpr (std::is_same_v<T, std::vector<double>>) {
                        os << std::fixed << std::setprecision(6);
                        for (const auto &val : vec)
                            os << val << " ";
                    } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                        for (const auto &val : vec)
                            os << val << " ";
                    } else {
                        os << "[Unsupported buffer type]";
                    }

                    if (total_elements > vec.size()) {
                        os << "... (" << (total_elements - vec.size()) << " more)";
                    }
                },
                buffer_variant);

        } catch (const std::exception &e) {
            os << "[Failed to unpack: " << e.what() << "]";
        }
        os << std::dec << "\n";
    } else if (view.data == nullptr) {
        os << " (null data pointer)";
    }

    return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TensorView &view) {
    std::ostringstream ss;
    ss << view;
    os << ss.str();
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const TensorAllocation &alloc) {
    os << "TensorAllocation(Shape: [";
    for (std::size_t i = 0; i < alloc.shape.size(); ++i) {
        os << alloc.shape[i] << (i < alloc.shape.size() - 1 ? ", " : "");
    }
    os << "], DType: " << alloc.dtype << ", Size: " << alloc.buffer.size() << " bytes)";

    if (!alloc.buffer.empty()) {
        os << "\n\t";
        try {
            const std::size_t byte_stride = get_size_bytes(alloc.dtype);
            if (byte_stride == 0)
                throw std::invalid_argument("Unknown byte size");

            const std::size_t total_elements = alloc.buffer.size() / byte_stride;
            const std::size_t display_count = std::min(total_elements, static_cast<std::size_t>(16));

            const std::vector head_buffer(alloc.buffer.begin(), alloc.buffer.begin() + (display_count * byte_stride));

            auto buffer_variant = unpack_buffer(head_buffer, alloc.dtype);

            std::visit(
                [&os, total_elements]<typename T0>(const T0 &vec) {
                    using T = std::decay_t<T0>;

                    if constexpr (std::is_same_v<T, std::vector<double>>) {
                        os << std::fixed << std::setprecision(6);
                        for (const auto &val : vec)
                            os << val << " ";
                    } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                        for (const auto &val : vec)
                            os << val << " ";
                    } else {
                        os << "[Unsupported buffer type]";
                    }

                    if (total_elements > vec.size()) {
                        os << "... (" << (total_elements - vec.size()) << " more)";
                    }
                },
                buffer_variant);

        } catch (const std::exception &e) {
            os << "[Failed to unpack: " << e.what() << "]";
        }

        os << std::dec << "\n";
    }
    return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TensorAllocation &alloc) {
    std::ostringstream ss;
    ss << alloc;
    os << ss.str();
    return os;
}

struct TensorConstraint {
    std::int64_t rank;
    std::vector<std::int64_t> dims;
    DataType expectedType;
    std::string debugName;

    [[nodiscard]] bool isStatic() const {
        for (const auto d : dims)
            if (d == -1)
                return false;
        return true;
    }
};

struct OutputAllocation {
    std::vector<std::vector<char>> buffers;
    std::vector<TensorView> views;
};
} // namespace conquer