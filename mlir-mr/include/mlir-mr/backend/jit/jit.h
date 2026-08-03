#include "llvm/IR/Module.h"

#include <memory>

int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule);
