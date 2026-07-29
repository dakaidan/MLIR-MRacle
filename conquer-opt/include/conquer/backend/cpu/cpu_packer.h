#pragma once

#include <cstdint>
#include <vector>

namespace conquer {
using MemRefDescriptor = std::vector<char>;

/// Helper to build a MemRef descriptor in a raw byte buffer.
/// Layout: [AllocatedPtr][AlignedPtr][Offset][Sizes...][Strides...]
/// All fields are 64-bit aligned (pointers are 64-bit on standard x64).
/// @param data Pointer to the data buffer.
/// @param shape The shape of the tensor.
/// @return A vector of bytes representing the MemRef descriptor.
MemRefDescriptor buildMemRefDescriptor(void *data, const std::vector<std::int64_t> &shape);
} // namespace conquer