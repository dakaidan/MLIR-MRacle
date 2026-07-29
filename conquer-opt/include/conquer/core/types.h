#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>

// TODO: We need to pack into these types when we start quantising the first layer
namespace conquer {
static_assert(std::endian::native == std::endian::little, "Only little-endian is supported");

enum class DataType {
    Unknown = 0,
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT32,
    INT16,
    INT8,
    // UINT8,
    INT4,  // Unpacked into int8 (1 byte stride)
    // UINT4, // Unpacked into uint8 (1 byte stride)
};

inline DataType from_string(const std::string &str) {
    std::string lower_str = str;
    for (auto &c : lower_str) c = std::tolower(c);

    if (lower_str == "fp32") return DataType::FP32;
    if (lower_str == "fp16") return DataType::FP16;
    if (lower_str == "bf16") return DataType::BF16;
    if (lower_str == "fp8_e4m3") return DataType::FP8_E4M3;
    if (lower_str == "fp8_e5m2") return DataType::FP8_E5M2;
    if (lower_str == "int32") return DataType::INT32;
    if (lower_str == "int16") return DataType::INT16;
    if (lower_str == "int8") return DataType::INT8;
    if (lower_str == "int4") return DataType::INT4;

    throw std::invalid_argument("Unknown DataType/Precision string: " + str);
}

inline std::string to_string(const DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return "FP32";
    case DataType::FP16:
        return "FP16";
    case DataType::BF16:
        return "BF16";
    case DataType::FP8_E4M3:
        return "FP8_E4M3";
    case DataType::FP8_E5M2:
        return "FP8_E5M2";
    case DataType::INT32:
        return "INT32";
    case DataType::INT16:
        return "INT16";
    case DataType::INT8:
        return "INT8";
    // case DataType::UINT8:
    //     return "UINT8";
    case DataType::INT4:
        return "INT4 (unpacked)";
    // case DataType::UINT4:
    //     return "UINT4 (unpacked)";
    default:
        return "Unknown";
    }
}

inline std::ostream &operator<<(std::ostream &os, const DataType &dtype) {
    os << to_string(dtype);
    return os;
}

template <DataType DT> struct UnpackTraits;

template <> struct UnpackTraits<DataType::FP32> {
    using ReturnType = double;
    static constexpr size_t size_bytes = 4;
};
template <> struct UnpackTraits<DataType::FP16> {
    using ReturnType = double;
    static constexpr size_t size_bytes = 2;
};
template <> struct UnpackTraits<DataType::BF16> {
    using ReturnType = double;
    static constexpr size_t size_bytes = 2;
};
template <> struct UnpackTraits<DataType::FP8_E4M3> {
    using ReturnType = double;
    static constexpr size_t size_bytes = 1;
};
template <> struct UnpackTraits<DataType::FP8_E5M2> {
    using ReturnType = double;
    static constexpr size_t size_bytes = 1;
};

template <> struct UnpackTraits<DataType::INT32> {
    using ReturnType = int64_t;
    static constexpr size_t size_bytes = 4;
};
template <> struct UnpackTraits<DataType::INT16> {
    using ReturnType = int64_t;
    static constexpr size_t size_bytes = 2;
};
template <> struct UnpackTraits<DataType::INT8> {
    using ReturnType = int64_t;
    static constexpr size_t size_bytes = 1;
};
// template <> struct UnpackTraits<DataType::UINT8> {
//     using ReturnType = int64_t;
//     static constexpr size_t size_bytes = 1;
// };
template <> struct UnpackTraits<DataType::INT4> {
    using ReturnType = int64_t;
    static constexpr size_t size_bytes = 1;
};
// template <> struct UnpackTraits<DataType::UINT4> {
//     using ReturnType = int64_t;
//     static constexpr size_t size_bytes = 1;
// };

template <typename T> T load_unaligned(const void *ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}

inline double float_bits_to_double(const uint32_t bits) { return std::bit_cast<float>(bits); }

template <DataType DT> typename UnpackTraits<DT>::ReturnType unpack(const void *ptr);

// ----------------------------------------------------------------------------
// INTEGER UNPACKING
// ----------------------------------------------------------------------------
template <> inline int64_t unpack<DataType::INT32>(const void *ptr) { return load_unaligned<int32_t>(ptr); }
template <> inline int64_t unpack<DataType::INT16>(const void *ptr) { return load_unaligned<int16_t>(ptr); }
template <> inline int64_t unpack<DataType::INT8>(const void *ptr) { return load_unaligned<int8_t>(ptr); }
// template <> inline int64_t unpack<DataType::UINT8>(const void *ptr) { return load_unaligned<uint8_t>(ptr); }

// template <> inline int64_t unpack<DataType::UINT4>(const void *ptr) { return load_unaligned<uint8_t>(ptr) & 0x0F; }

template <> inline int64_t unpack<DataType::INT4>(const void *ptr) {
    auto val = load_unaligned<int8_t>(ptr);
    val &= 0x0F;
    if (val & 0x08)
        val |= 0xF0;
    return val;
}

// ----------------------------------------------------------------------------
// FLOATING POINT UNPACKING
// ----------------------------------------------------------------------------
template <> inline double unpack<DataType::FP32>(const void *ptr) { return load_unaligned<float>(ptr); }

template <> inline double unpack<DataType::BF16>(const void *ptr) {
    const uint32_t val = static_cast<uint32_t>(load_unaligned<uint16_t>(ptr)) << 16;
    return float_bits_to_double(val);
}

template <> inline double unpack<DataType::FP16>(const void *ptr) {
    const auto h = load_unaligned<uint16_t>(ptr);
    const uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF);

    if (exp == 0) {
        if (frac == 0)
            return float_bits_to_double(sign); // Zero
        exp = 1;
        while (!(frac & 0x0400)) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x03FF;
        exp = exp - 15 + 127;
        return float_bits_to_double(sign | (exp << 23) | (frac << 13));
    }
    if (exp == 0x1F) { // Inf or NaN
        return float_bits_to_double(sign | 0x7F800000 | (frac << 13));
    }

    exp = exp - 15 + 127;
    return float_bits_to_double(sign | (exp << 23) | (frac << 13));
}

template <> inline double unpack<DataType::FP8_E5M2>(const void *ptr) {
    const uint8_t b = load_unaligned<uint8_t>(ptr);
    const uint32_t sign = (b & 0x80) << 24;
    uint32_t exp = (b & 0x7C) >> 2;
    uint32_t frac = (b & 0x03);

    if (exp == 0) {
        if (frac == 0)
            return float_bits_to_double(sign); // Zero
        exp = 1;
        while (!(frac & 0x04)) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x03;
        exp = exp - 15 + 127;
        return float_bits_to_double(sign | (exp << 23) | (frac << 21));
    }
    if (exp == 0x1F) { // Inf or NaN
        return float_bits_to_double(sign | 0x7F800000 | (frac << 21));
    }

    exp = exp - 15 + 127;
    return float_bits_to_double(sign | (exp << 23) | (frac << 21));
}

template <> inline double unpack<DataType::FP8_E4M3>(const void *ptr) {
    const auto b = load_unaligned<uint8_t>(ptr);
    const uint32_t sign = (b & 0x80) << 24;
    uint32_t exp = (b & 0x78) >> 3;
    uint32_t frac = (b & 0x07);

    if (exp == 0) {
        if (frac == 0)
            return float_bits_to_double(sign); // Zero
        exp = 1;
        while (!(frac & 0x08)) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x07;
        exp = exp - 7 + 127;
        return float_bits_to_double(sign | (exp << 23) | (frac << 20));
    }
    if (exp == 15 && frac == 7) {
        return float_bits_to_double(sign | 0x7F800000 | (frac << 20));
    }

    exp = exp - 7 + 127;
    return float_bits_to_double(sign | (exp << 23) | (frac << 20));
}

// ----------------------------------------------------------------------------
// VECTOR UNPACKING
// ----------------------------------------------------------------------------
using UnpackedBuffer = std::variant<std::vector<double>, std::vector<int64_t>>;

template <DataType DT>
std::vector<typename UnpackTraits<DT>::ReturnType> unpack_typed_vector(const std::vector<char> &buffer) {
    using RType = typename UnpackTraits<DT>::ReturnType;
    constexpr size_t stride = UnpackTraits<DT>::size_bytes;

    const size_t element_count = buffer.size() / stride;
    std::vector<RType> result(element_count);

    const char *ptr = buffer.data();
    for (size_t i = 0; i < element_count; ++i) {
        result[i] = unpack<DT>(ptr + (i * stride));
    }

    return result;
}

inline UnpackedBuffer unpack_buffer(const std::vector<char> &buffer, const DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return unpack_typed_vector<DataType::FP32>(buffer);
    case DataType::FP16:
        return unpack_typed_vector<DataType::FP16>(buffer);
    case DataType::BF16:
        return unpack_typed_vector<DataType::BF16>(buffer);
    case DataType::FP8_E4M3:
        return unpack_typed_vector<DataType::FP8_E4M3>(buffer);
    case DataType::FP8_E5M2:
        return unpack_typed_vector<DataType::FP8_E5M2>(buffer);
    case DataType::INT32:
        return unpack_typed_vector<DataType::INT32>(buffer);
    case DataType::INT16:
        return unpack_typed_vector<DataType::INT16>(buffer);
    case DataType::INT8:
        return unpack_typed_vector<DataType::INT8>(buffer);
    // case DataType::UINT8:
    //     return unpack_typed_vector<DataType::UINT8>(buffer);
    case DataType::INT4:
        return unpack_typed_vector<DataType::INT4>(buffer);
    // case DataType::UINT4:
    //     return unpack_typed_vector<DataType::UINT4>(buffer);
    default:
        throw std::invalid_argument("Cannot unpack: Unknown DataType");
    }
}

template <DataType DT> constexpr size_t get_size_bytes() { return UnpackTraits<DT>::size_bytes; }

constexpr size_t get_size_bytes(const DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
    case DataType::INT32:
        return 4;

    case DataType::FP16:
    case DataType::BF16:
    case DataType::INT16:
        return 2;

    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::INT8:
    // case DataType::UINT8:
    case DataType::INT4:  // 1 byte stride assumed (unpacked into int8)
    // case DataType::UINT4: // 1 byte stride assumed
        return 1;

    case DataType::Unknown:
        return 0;
    }
    return 0;
}
} // namespace conquer