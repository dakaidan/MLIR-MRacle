#pragma once
#include "conquer/loaders/data/data.h"

namespace conquer {
    class BinaryDataLoader : public DataLoader {
    public:
        // Note: For a raw binary, we don't inherently know the shape/type.
        // For right now, we can inject a predefined shape/type.
        std::vector<int64_t> expected_shape;
        DataType expected_dtype = DataType::FP32;

        std::vector<TensorAllocation> loadData(const std::string &dataPath) override;

        [[nodiscard]] bool supportsExtension(const std::string &extension) const override;
    };
} // namespace conquer