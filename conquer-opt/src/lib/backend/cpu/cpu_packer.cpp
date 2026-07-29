#include "conquer/backend/cpu/cpu_packer.h"

#include <cstdint>
#include <cstring>

conquer::MemRefDescriptor conquer::buildMemRefDescriptor(void *data, const std::vector<std::int64_t> &shape) {
    const int64_t rank = shape.size();

    // Calculate strides (assuming standard row-major contiguous layout)
    std::vector<int64_t> strides(rank);
    if (rank > 0) {
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // Calculate total size of the descriptor structure
    // 2 ptrs + 1 int64 (offset) + rank * int64 (sizes) + rank * int64 (strides)
    const size_t size = sizeof(void *) * 2 + sizeof(int64_t) * (1 + 2 * rank);
    std::vector<char> descriptor(size);
    char *ptr = descriptor.data();

    // 1. Allocated Pointer
    std::memcpy(ptr, &data, sizeof(void *));
    ptr += sizeof(void *);

    // 2. Aligned Pointer
    std::memcpy(ptr, &data, sizeof(void *));
    ptr += sizeof(void *);

    // 3. Offset (0)
    constexpr int64_t offset = 0;
    std::memcpy(ptr, &offset, sizeof(int64_t));
    ptr += sizeof(int64_t);

    // 4. Sizes
    if (rank > 0) {
        std::memcpy(ptr, shape.data(), rank * sizeof(int64_t));
        ptr += rank * sizeof(int64_t);
    }

    // 5. Strides
    if (rank > 0) {
        std::memcpy(ptr, strides.data(), rank * sizeof(int64_t));
    }

    return descriptor;
}
