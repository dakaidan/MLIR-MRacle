#include "conquer/passes/analysis/strip_annotations_pass.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-strip-annotations-pass"

namespace {
    constexpr llvm::StringRef annotationPrefixes[] = {
    "conquer.",
    "weight_stats.",
    "weight_use.",
    "activation_stats."
};

static bool isAnnotationAttr(const llvm::StringRef attrName) {
    for (const llvm::StringRef prefix : annotationPrefixes) {
        if (attrName.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

} // namespace

void conquer::StripAnnotationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();

    module.walk([](mlir::Operation *op) {
        // 1. Strip standard operation attributes
        // We collect attributes to remove first to avoid invalidating the iterator while traversing.
        llvm::SmallVector<mlir::StringAttr> attrsToRemove;
        for (const mlir::NamedAttribute attr : op->getAttrs()) {
            const llvm::StringRef attrName = attr.getName().getValue();
            if (isAnnotationAttr(attrName)) {
                attrsToRemove.push_back(attr.getName());
            }
        }

        for (const mlir::StringAttr attrName : attrsToRemove) {
            L_TRACE("Removing annotation: " << attrName.getValue()
                                    << " from op: " << compact(op));
            op->removeAttr(attrName);
        }

        // 2. Strip function argument attributes
        // Operations like func.func hold argument attributes in a separate array.
        if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
                llvm::SmallVector<mlir::StringAttr> argAttrsToRemove;

                if (mlir::DictionaryAttr argDict = funcOp.getArgAttrDict(i)) {
                    for (const mlir::NamedAttribute attr : argDict) {
                        if (isAnnotationAttr(attr.getName().getValue())) {
                            argAttrsToRemove.push_back(attr.getName());
                        }
                    }
                }

                for (const mlir::StringAttr attrName : argAttrsToRemove) {
                    L_TRACE("Removing arg annotation: " << attrName.getValue()
                                            << " from arg: " << i << " in func: " << funcOp.getName());
                    funcOp.removeArgAttr(i, attrName);
                }
            }
        }
    });
}