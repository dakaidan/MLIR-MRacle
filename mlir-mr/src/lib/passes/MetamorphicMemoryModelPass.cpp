#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {

#define GEN_PASS_DECL_METAMORPHICMEMORYMODELPASS
#include "MetamorphicMemoryModelPass.inc"

}

namespace mlir {

#define GEN_PASS_DEF_METAMORPHICMEMORYMODELPASS
#include "MetamorphicMemoryModelPass.inc"

struct MetamorphicMemoryModelPass
    : impl::MetamorphicMemoryModelPassBase<MetamorphicMemoryModelPass> {
  using impl::MetamorphicMemoryModelPassBase<MetamorphicMemoryModelPass>::MetamorphicMemoryModelPassBase;

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    // TODO: metamorphic transformations
  }
};

std::unique_ptr<Pass> createMetamorphicMemoryModelPass() {
  return std::make_unique<MetamorphicMemoryModelPass>();
}

}

#define GEN_PASS_REGISTRATION
#include "MetamorphicMemoryModelPass.inc"