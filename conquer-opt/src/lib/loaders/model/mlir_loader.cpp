#include "conquer/loaders/model/mlir_loader.h"

#include "conquer/core/context.h"

#include <mlir/Parser/Parser.h>
#include <llvm/Support/MemoryBuffer.h>

#include <stdexcept>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-mlir-loader"

mlir::OwningOpRef<mlir::Operation *> conquer::MLIRModelLoader::loadModel(const std::string &modelPath, mlir::MLIRContext *ctx) {
    const llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(modelPath);

    if (const std::error_code ec = fileOrErr.getError()) {
        throw std::runtime_error("Could not open input file: " + ec.message());
    }

    const auto buffer = fileOrErr.get()->getBuffer();
    const auto parserConfig = mlir::ParserConfig(ctx);
    auto module = mlir::parseSourceString(buffer, parserConfig);

    if (!module) {
        throw std::runtime_error("Failed to parse MLIR module from file: " + modelPath);
    }

    return module;
}

bool conquer::MLIRModelLoader::supportsExtension(const std::string &extension) const {
    return extension == "mlir";
}

bool conquer::MLIRModelLoader::supportsFormat(const std::string &modelPath,
                                     mlir::MLIRContext *ctx) const {
    const llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(modelPath);

    if (fileOrErr.getError()) {
        return false;
    }

    const auto buffer = fileOrErr.get()->getBuffer();
    const auto parserConfig = mlir::ParserConfig(ctx);
    const auto module = mlir::parseSourceString(buffer, parserConfig);

    return static_cast<bool>(module);
}