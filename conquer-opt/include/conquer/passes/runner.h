#pragma once

#include "conquer/core/logging.h"
#include <mlir/Pass/PassManager.h>

#include <llvm/Support/Debug.h>

#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-pass-manager"

namespace conquer {
/// A utility function for running a single MLIR pass on a module.
/// @tparam PassType The type of the pass to run. Must have a constructor matching the provided Args.
/// @param module The MLIR module to run the pass on.
/// @param args Optional arguments perfectly forwarded to the PassType constructor.
template <typename PassType, typename... Args> void runPassOnModule(mlir::Operation *module, Args &&...args) {
    // While we probably could use global context, this should be preferred
    // in case of special contexts needed by certain modules
    mlir::PassManager pm(module->getContext());
    auto pass = std::make_unique<PassType>(std::forward<Args>(args)...);
    const auto passArg = pass->getArgument();
    pm.addPass(std::move(pass));

    pm.enableCrashReproducerGeneration("crash.mlir");

    L_DEBUG("Running pass '" << passArg << "' on module.");
    if (mlir::failed(pm.run(module))) {
        L_DEBUG("Pass '" << passArg << "' failed.");
        throw std::runtime_error("Pass pipeline failed!");
    }
    L_DEBUG("Pass '" << passArg << "' succeeded.");
}
} // namespace conquer
