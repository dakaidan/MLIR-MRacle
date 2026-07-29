#pragma once

#include "conquer/loaders/data/data.h"
#include "conquer/loaders/model/model.h"
#include "conquer/loaders/model/mlir_loader.h"
#include "conquer/loaders/data/npy_loader.h"
#include "conquer/loaders/data/binary_loader.h"
#include "conquer/core/logging.h"

#include <llvm/Support/Debug.h>
#include <filesystem>
#include <vector>
#include <memory>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-loader"

namespace conquer {

/// Manages a collection of available model loaders. This class is responsible for selecting the
/// appropriate loader for a given model file based on its extension or format. It follows a
/// singleton pattern, with a single instance accessible via `getModelLoaderManager()`.
class ModelLoaderManager {
  public:
    ModelLoaderManager() {
        registerLoader(std::make_shared<MLIRModelLoader>());
    }

    void registerLoader(const std::shared_ptr<ModelLoader> &loader) {
        loaders.push_back(loader);
    }

    /// Finds and returns a suitable model loader for the given model file.
    /// The selection process is as follows:
    /// 1. Try to find a loader that supports the model's file extension.
    /// 2. If no extension-based loader is found, fall back to checking the model's format by
    ///    asking each loader to inspect the file's content.
    /// @param modelPath The path to the model file.
    /// @return A shared pointer to the selected `ModelLoader`.
    /// @throws std::runtime_error if no suitable loader is found.
    [[nodiscard]] std::shared_ptr<ModelLoader> getLoader(const std::string &modelPath, mlir::MLIRContext *ctx) const {
        const std::string extension = getFileExtension(modelPath);
        L_DEBUG("Attempting to find loader for model: " << modelPath << " (extension: " << extension << ")");

        for (const auto &loader : loaders) {
            if (loader->supportsExtension(extension)) {
                L_DEBUG("Found loader based on file extension.");
                return loader;
            }
        }

        L_DEBUG("No loader found for extension, trying format detection.");
        for (const auto &loader : loaders) {
            if (loader->supportsFormat(modelPath, ctx)) {
                L_DEBUG("Found loader based on file format.");
                return loader;
            }
        }

        throw std::runtime_error("No suitable ModelLoader found for: " + modelPath);
    }

  private:
    std::vector<std::shared_ptr<ModelLoader>> loaders;

    static std::string getFileExtension(const std::string &filename) {
        std::string ext = std::filesystem::path(filename).extension().string();
        if (ext.starts_with('.')) {
            ext = ext.substr(1);
        }
        return ext;
    }
};

inline ModelLoaderManager &getModelLoaderManager() {
    static ModelLoaderManager instance;
    return instance;
}

class DataLoaderManager {
  public:
    DataLoaderManager() {
        registerLoader(std::make_shared<NpyDataLoader>());
        registerLoader(std::make_shared<BinaryDataLoader>());
    }

    void registerLoader(const std::shared_ptr<DataLoader> &loader) { loaders.push_back(loader); }

    [[nodiscard]] std::shared_ptr<DataLoader> getLoader(const std::string &dataPath) const {
        std::string ext = std::filesystem::path(dataPath).extension().string();
        if (ext.starts_with('.'))
            ext = ext.substr(1);

        for (const auto &loader : loaders) {
            if (loader->supportsExtension(ext))
                return loader;
        }
        throw std::runtime_error("No DataLoader found for extension: " + ext);
    }

  private:
    std::vector<std::shared_ptr<DataLoader>> loaders;
};

inline DataLoaderManager &getDataLoaderManager() {
    static DataLoaderManager instance;
    return instance;
}
} // namespace conquer