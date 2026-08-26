#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "mlir-mracle/passes/transforms/Transforms.h"
#include "mlir-mracle/context/context.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <utility>

namespace mlir {
namespace metamorphic {
namespace {

using TransformRegistry =
    llvm::SmallVector<std::unique_ptr<MetamorphicTransform>, 32>;

TransformRegistry &getRegistry() {
    static TransformRegistry registry = [] {
        TransformRegistry reg;
        for (const FunctionTransform &t : getGenericTransforms())
            reg.push_back(std::make_unique<FunctionTransform>(t));
        for (const FunctionTransform &t : getArmTransforms())
            reg.push_back(std::make_unique<FunctionTransform>(t));
        return reg;
    }();
    return registry;
}

} // namespace

void registerTransform(std::unique_ptr<MetamorphicTransform> transform) {
    TransformRegistry &registry = getRegistry();
    for (auto &entry : registry)
        if (entry->getName() == transform->getName()) {
            entry = std::move(transform);
            return;
        }
    registry.push_back(std::move(transform));
}

llvm::SmallVector<const MetamorphicTransform *, 32>
getTransforms(llvm::StringRef target) {
    llvm::SmallVector<const MetamorphicTransform *, 32> result;
    for (const auto &entry : getRegistry())
        if (entry->getTarget() == "generic" || entry->getTarget() == target)
            result.push_back(entry.get());
    return result;
}

mlir_mracle::OutcomeRelation composeRelation(mlir_mracle::OutcomeRelation a,
                                             mlir_mracle::OutcomeRelation b) {
    if (a == mlir_mracle::OutcomeRelation::Equality)
        return b;
    if (b == mlir_mracle::OutcomeRelation::Equality)
        return a;
    if (a == b)
        return a;
    llvm_unreachable("mixing subset and superset transforms in one run");
}

bool canApplyAfter(mlir_mracle::OutcomeRelation cur,
                   mlir_mracle::OutcomeRelation next) {
    if (cur == mlir_mracle::OutcomeRelation::Equality)
        return true;
    return cur == next;
}

} // namespace metamorphic
} // namespace mlir
