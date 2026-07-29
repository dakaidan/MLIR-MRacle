#pragma once
#include "conquer/core/types.h"

#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace conquer {
enum class DTypeSupport { Native, Emulated, Unsupported };

inline std::string to_string(const DTypeSupport support) {
    switch (support) {
    case DTypeSupport::Native:
        return "Native";
    case DTypeSupport::Emulated:
        return "Emulated";
    case DTypeSupport::Unsupported:
        return "Unsupported";
    }
    return "Unknown";
}

inline std::ostream &operator<<(std::ostream &os, const DTypeSupport support) {
    os << to_string(support);
    return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DTypeSupport support) {
    os << to_string(support);
    return os;
}

struct DTypeCapability {
    DataType dtype;
    DTypeSupport support;
};

inline std::string to_string(const DTypeCapability &cap) {
    return to_string(cap.dtype) + ": " + to_string(cap.support);
}

inline std::ostream &operator<<(std::ostream &os, const DTypeCapability &cap) {
    os << to_string(cap);
    return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DTypeCapability &cap) {
    os << to_string(cap);
    return os;
}

struct HardwareCapability {
    std::string device_name;
    std::string device_uri;
    std::vector<DTypeCapability> dtypes;

    [[nodiscard]] DTypeSupport query(const DataType dt) const {
        for (const auto &[dtype, support] : dtypes)
            if (dtype == dt)
                return support;
        return DTypeSupport::Unsupported;
    }

    [[nodiscard]] bool is_native(const DataType dt) const { return query(dt) == DTypeSupport::Native; }
    [[nodiscard]] bool is_usable(const DataType dt) const { return query(dt) != DTypeSupport::Unsupported; }

    [[nodiscard]] std::vector<DataType> native_types() const {
        std::vector<DataType> out;
        for (const auto &[dtype, support] : dtypes)
            if (support == DTypeSupport::Native)
                out.push_back(dtype);
        return out;
    }

    [[nodiscard]] std::vector<DataType> usable_types() const {
        std::vector<DataType> out;
        for (const auto &[dtype, support] : dtypes)
            if (support != DTypeSupport::Unsupported)
                out.push_back(dtype);
        return out;
    }
};

inline std::string to_string(const HardwareCapability &cap) {
    std::string out = "Device: " + cap.device_name + " (" + cap.device_uri + ")\n";
    out += "Supported Data Types:\n";
    for (const auto &[dtype, support] : cap.dtypes)
        out += "  - " + to_string(dtype) + ": " + to_string(support) + "\n";
    return out;
}

inline std::ostream &operator<<(std::ostream &os, const HardwareCapability &cap) {
    os << to_string(cap);
    return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HardwareCapability &cap) {
    os << to_string(cap);
    return os;
}

HardwareCapability query_cpu_capability();

/// Derives CUDA dtype support querying NVML for the compute capability.
HardwareCapability query_cuda_capability();

/// Derives Vulkan dtype support querying the Vulkan API.
HardwareCapability query_vulkan_capability();

/// Derives ROCM dtype support querying HIP for the architecture string.
HardwareCapability query_rocm_capability();
} // namespace conquer
