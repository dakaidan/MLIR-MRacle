#include "mlir-mr/context/context.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

namespace mlir_mr {

MLIRSetup::MLIRSetup() : pm(&mlirContext) {
    mlir::registerBuiltinDialectTranslation(mlirContext);
    mlir::registerLLVMDialectTranslation(mlirContext);

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerOpenMPDialectTranslation(registry);
    mlirContext.appendDialectRegistry(registry);
}

} // namespace mlir_mr
