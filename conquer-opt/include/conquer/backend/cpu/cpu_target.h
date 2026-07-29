#pragma once

#include "conquer/backend/target.h"
#include "conquer/runtime/session.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include <vector>
#include <mutex>

namespace conquer {

  class CPUTarget : public Target {
  public:
    CPUTarget() = default;
    ~CPUTarget() override = default;

    /// Builds and runs the pass pipeline to lower TOSA to the LLVM dialect.
    /// It also stores the reference to the module for the subsequent execute call.
    /// @param module The MLIR module to lower. (should pass a copy as we take ownership)
    llvm::Error compile(mlir::Operation *module) override;

    llvm::Error compile(llvm::StringRef bytecode) override;

    llvm::Error save_compiled_module_to_file(const std::string &filename) const override;
    llvm::Error load_compiled_module_from_file(const std::string &filename) override;

    /// Creates an MLIR ExecutionEngine, JIT compiles the stored module,
    /// and invokes the "main" function with the provided buffers.
    llvm::Error execute(const std::vector<TensorView> &inputs, const std::vector<TensorView> &outputs) override;

    [[nodiscard]] HardwareCapability query_capability() const override;

  private:
    llvm::Error build_execution_engine();
    /// We hold a reference to the module because the `execute` method
    /// relies on the state produced by `lower`.
    mlir::OwningOpRef<mlir::Operation *> loweredModule;
    std::unique_ptr<mlir::MLIRContext> local_context_;
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    std::once_flag engine_init_flag_;
  };

} // namespace conquer
