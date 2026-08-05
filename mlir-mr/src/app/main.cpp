#include "mlir-mr/context/context.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>   // for strcmp

int main(int argc, char **argv) {

    // parsing for the --print-mlir flag
    bool printMLIR = false;
    int newArgc = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-mlir") == 0) {
            printMLIR = true;
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (argc < 2) {
        llvm::errs() << "Usage: mlir-mr-opt [--print-mlir] <path-to-mlir-file>\n";
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

    if (mlir::failed(setup.pm.run(*module)))
        return 1;

    // Print MLIR if flagged
    if (printMLIR)
        module->print(llvm::outs());

    if (mlir::failed(mlir_mr::lowerToLLVM(*module, &setup.mlirContext)))
        return 1;

    std::string outPath =
        mlir_mr::translateAndWriteToFile(*module, setup.llvmContext, "output.ll");
    if (outPath.empty())
        return 1;

    llvm::outs() << "Wrote output to " << outPath << "\n";
    return 0;
}