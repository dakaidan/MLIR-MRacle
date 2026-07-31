#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"

#include "mlir-mr/backend/jit.h"

#include <iostream>
#include <string>
#include <utility>

int main(int argc, char **argv) {

    // init LLVM, MLIR context, and pass manager
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;

    mlir::registerBuiltinDialectTranslation(mlirContext);
    mlir::registerLLVMDialectTranslation(mlirContext);

    // Load dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlirContext.appendDialectRegistry(registry);

    // dialects must be loaded into context before pass manager init
    mlir::PassManager pm(&mlirContext);

    // Argument handling
    if (argc < 2) {
        llvm::errs() << "Usage: mlir-mr-opt <path-to-mlir-file>\n";
        return 1;
    }
    std::string inputPath = argv[1];

    // Parse into MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputPath, &mlirContext);
    if (!module) {
        llvm::errs() << "Parse error\n";
        return 1;
    }

    mlir::ConversionTarget target(mlirContext);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&mlirContext);

    // Many conversion passes
    // TODO: consider how to manager our own conversion passes and patterns
    // I guess we can just do it manually before this
    mlir::RewritePatternSet patterns(&mlirContext);
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyFullConversion(*module, target, std::move(patterns))))
        return 1;

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Translate MLIR module to LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "LLVM IR translation failed\n";
        return 1;
    }

    int ret = executeLLVMModuleWithJIT(std::move(llvmModule));
    std::cout << "Execution returned: " << ret << "\n";

    return 0;
}