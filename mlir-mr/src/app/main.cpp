#include "mlir-mr/context/context.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>   // for strtol
#include <cstring>   // for strcmp
#include <string>

int main(int argc, char **argv) {

    // parse control flags and strip them from argv
    bool printMLIR = false;
    bool debug = false;
    int seed = 42;
    std::string transforms;
    int newArgc = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-mlir") == 0) {
            printMLIR = true;
        } else if (std::strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            seed = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--transforms=", 13) == 0) {
            transforms = argv[i] + 13;
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (argc < 2) {
        llvm::errs() << "Usage: mlir-mr-opt [--print-mlir] [--debug] [--seed=N] "
                        "[--transforms=NAME[,NAME...]] <path-to-mlir-file>\n";
        return 1;
    }

    // Setup context
    mlir_mr::MLIRSetup setup(seed, transforms, debug);

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