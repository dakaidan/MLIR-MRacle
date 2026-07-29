#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <string>

namespace conquer {
/// An abstract interface for a class that can load a model from a given format
/// into an in-memory MLIR representation. This allows ConQuER to be extended
/// to support various model formats (e.g., ONNX, TensorFlow SavedModel) by
/// providing a concrete implementation of this interface.
class ModelLoader {
  public:
    virtual ~ModelLoader() = default;

    /// Loads a model from the given path and returns it as an MLIR module.
    /// @param modelPath The path to the model file.
    /// @return An owning reference to the top-level MLIR operation (the module).
    virtual mlir::OwningOpRef<mlir::Operation *> loadModel(const std::string &modelPath, mlir::MLIRContext *ctx) = 0;

    /// Checks if this loader supports a given file extension.
    /// @param extension The file extension (without the leading dot).
    /// @return True if the loader supports the extension, false otherwise.
    virtual bool supportsExtension(const std::string &extension) const = 0;

    /// A fallback mechanism to check if the loader supports a given model format
    /// by inspecting the file's content. This is used when the file extension
    /// does not match any known loader.
    /// @param modelPath The path to the model file.
    /// @return True if the loader can handle the model format, false otherwise.
    virtual bool supportsFormat(const std::string &modelPath, mlir::MLIRContext *ctx) const = 0;
};
} // namespace conquer