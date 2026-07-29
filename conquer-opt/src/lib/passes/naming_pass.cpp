#include "conquer/passes/naming_pass.h"
#include "conquer/core/logging.h"

#include "conquer/quantisation/tosa_utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Debug.h>

#include <map>
#include <string>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-naming"

void conquer::ModelNamingPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(&getContext());

    // A map to keep track of the number of times we've seen each operation type.
    // This is used to generate unique names.
    std::map<std::string, int> opCounters;

    module.walk([&](mlir::Operation *op) {
        // We don't want to name the top-level module or terminator operations.
        if (llvm::isa<mlir::ModuleOp>(op) || op->hasTrait<mlir::OpTrait::IsTerminator>())
            return;

        // Skip if it is not a quantisable operation
        if (!isQuantisable(op))
            return;

        // Create a base name for the operation from its MLIR name.
        // We replace dots with underscores to make the names more friendly for
        // things like file names or variable names.
        std::string name = op->getName().getStringRef().str();
        std::ranges::replace(name, '.', '_');

        // Generate a unique name by appending a counter.
        const int index = opCounters[name]++;
        const std::string uniqueName = name + "_" + std::to_string(index);

        L_TRACE("Assigning name: " << uniqueName << " to op: " << compact(op));
        op->setAttr("conquer.name", builder.getStringAttr(uniqueName));
    });
}