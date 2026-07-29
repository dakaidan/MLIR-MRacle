#include "conquer/loaders/data/binary_loader.h"
#include "conquer/loaders/data/data.h"
#include "conquer/loaders/manager.h"

#include <fstream>
#include <stdexcept>

std::vector<conquer::TensorAllocation> conquer::BinaryDataLoader::loadData(const std::string &dataPath) {
    std::ifstream file(dataPath, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open data file: " + dataPath);

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    TensorAllocation alloc;
    alloc.shape = expected_shape;
    alloc.dtype = expected_dtype;
    alloc.buffer.resize(size);

    if (!file.read(alloc.buffer.data(), size)) {
        throw std::runtime_error("Failed to read binary data.");
    }

    return {std::move(alloc)};
}

[[nodiscard]] bool conquer::BinaryDataLoader::supportsExtension(const std::string &extension) const { return extension == "bin"; }
