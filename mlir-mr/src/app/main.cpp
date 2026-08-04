#include "mlir-mr/context/context.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        llvm::errs() << "Usage: mlir-mr-opt <path-to-mlir-file>\n";
        return 1;
    }

    // Setup context
    mlir_mr::MLIRSetup setup;

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &setup.mlirContext);
    if (!module) {
        llvm::errs() << "Parse error\n";
        return 1;
    }

    if (mlir::failed(mlir_mr::lowerToLLVM(*module, &setup.mlirContext)))
        return 1;

    // TODO: add flag depending on the tool?
    std::string outPath =
        mlir_mr::translateAndWriteToFile(*module, setup.llvmContext, "output.ll");
    if (outPath.empty())
        return 1;

    llvm::outs() << "Wrote output to " << outPath << "\n";
    return 0;
}