#pragma once

#include "conquer/backend/capabilities.h"
#include "conquer/runtime/session.h"

#include <llvm/Support/Error.h>

#include <vector>

namespace conquer {
    /// Abstract base class for hardware execution targets.
    class Target {
    public:
        virtual ~Target() = default;

        /// Configures and runs the lowering pipeline to get the module ready for this target.
        /// For CPU, this lowers TOSA -> Linalg -> LLVM Dialect.
        /// @param module The MLIR module to lower. (should pass a copy as we take ownership)
        virtual llvm::Error compile(mlir::Operation *module) = 0;

        /// Compiles with the bytecode string, so iscolated
        virtual llvm::Error compile(llvm::StringRef bytecode) = 0;

        virtual llvm::Error save_compiled_module_to_file(const std::string &filename) const = 0;
        virtual llvm::Error load_compiled_module_from_file(const std::string &filename) = 0;

        /// JIT compiles (if necessary) and executes the default entry point of the module.
        /// @param inputs A vector of input TensorViews.
        /// @param outputs A vector of output TensorViews.
        /// @return An llvm::Error indicating success or failure.
        virtual llvm::Error execute(const std::vector<TensorView> &inputs, const std::vector<TensorView> &outputs) = 0;

        [[nodiscard]] virtual HardwareCapability query_capability() const = 0;
    };

} // namespace conquer