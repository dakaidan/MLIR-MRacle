#include "mlir-mr/io/io.h"

#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <system_error>

namespace mlir_mr {

std::string translateAndWriteToFile(mlir::ModuleOp module,
                                    llvm::LLVMContext &llvmContext,
                                    const std::string &filename) {
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "LLVM IR translation failed\n";
        return "";
    }

    std::filesystem::path outDir = "test-outputs";
    std::filesystem::create_directories(outDir);
    std::filesystem::path outPath = outDir / filename;

    std::error_code ec;
    llvm::raw_fd_ostream os(outPath.string(), ec);
    if (ec) {
        llvm::errs() << "Error opening " << outPath << ": " << ec.message() << "\n";
        return "";
    }

    llvmModule->print(os, nullptr);
    return outPath.string();
}

} // namespace mlir_mr
