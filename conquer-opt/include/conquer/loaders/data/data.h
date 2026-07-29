#pragma once
#include "conquer/runtime/session.h"

#include <string>
#include <vector>

namespace conquer {
class DataLoader {
  public:
    virtual ~DataLoader() = default;

    /// Loads data from a file. Returns a vector because some formats (like .npz or HDF5)
    /// might contain multiple input tensors.
    virtual std::vector<TensorAllocation> loadData(const std::string &dataPath) = 0;

    virtual bool supportsExtension(const std::string &extension) const = 0;
};
} // namespace conquer