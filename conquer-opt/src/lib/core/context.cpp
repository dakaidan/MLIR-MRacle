#include "conquer/core/context.h"
#include "conquer/dialect/ConquerDialect.h"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>


#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-context"

namespace conquer {

mlir::DialectRegistry makeDialectRegistry() {
    mlir::DialectRegistry registry;

    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    registry.insert<ConquerDialect>();

    // External models / interfaces used by your lowering pipeline
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);

    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    return registry;
}

std::unique_ptr<mlir::MLIRContext> createMLIRContext() {
    auto ctx = std::make_unique<mlir::MLIRContext>();
    auto registry = makeDialectRegistry();
    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();
    return ctx;
}
} // namespace conquer
