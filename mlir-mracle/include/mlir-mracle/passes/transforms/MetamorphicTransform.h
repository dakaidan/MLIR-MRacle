#pragma once

#include "mlir-mracle/core/types.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <random>

namespace mlir {
namespace metamorphic {

// A metamorphic transform. Transforms are objects, so the outcome relation
// is a virtual property: external code can override it by subclassing (see
// FunctionTransform) and re-registering the instance with registerTransform(),
// without touching the built-in transform implementations.
class MetamorphicTransform {
public:
    virtual ~MetamorphicTransform() = default;

    virtual llvm::StringRef getName() const = 0;
    // Hardware/memory-model target this transform is valid for:
    //   "generic" — valid on every target and always included
    //   "armv8"   — ARMv8-A weak memory model
    //   "tso"     — x86 TSO (future)
    virtual llvm::StringRef getTarget() const = 0;
    virtual bool apply(func::FuncOp, RewriterBase &, std::mt19937 &) const = 0;
    virtual mlir_mracle::OutcomeRelation getRelation() const = 0;
};

// Concrete transform wrapping a free-function implementation plus its
// default outcome relation. Subclass it, override getRelation() (or
// apply()), and hand the instance to registerTransform() to override a
// built-in transform externally.
class FunctionTransform : public MetamorphicTransform {
public:
    using TransformFn = bool (*)(func::FuncOp, RewriterBase &, std::mt19937 &);

    FunctionTransform(llvm::StringRef name, llvm::StringRef target,
                      TransformFn fn, mlir_mracle::OutcomeRelation relation)
        : name(name), target(target), fn(fn), relation(relation) {}

    llvm::StringRef getName() const override { return name; }
    llvm::StringRef getTarget() const override { return target; }
    bool apply(func::FuncOp op, RewriterBase &rewriter,
               std::mt19937 &rng) const override {
        return fn(op, rewriter, rng);
    }
    mlir_mracle::OutcomeRelation getRelation() const override {
        return relation;
    }

protected:
    llvm::StringRef name;
    llvm::StringRef target;
    TransformFn fn;
    mlir_mracle::OutcomeRelation relation;
};

// Registers a transform, replacing any built-in transform with the same name.
void registerTransform(std::unique_ptr<MetamorphicTransform> transform);

// All transforms valid for `target`: every "generic" transform is always
// included, plus any transform whose target equals `target`. The --model
// pass option selects the target; an empty model selects the generic
// transforms alone.
llvm::SmallVector<const MetamorphicTransform *, 32>
getTransforms(llvm::StringRef target);

// Compose two applied relations in application order.
mlir_mracle::OutcomeRelation composeRelation(mlir_mracle::OutcomeRelation a,
                                             mlir_mracle::OutcomeRelation b);

// True if a transform of relation `next` may follow one of relation `cur`.
bool canApplyAfter(mlir_mracle::OutcomeRelation cur,
                   mlir_mracle::OutcomeRelation next);

} // namespace metamorphic
} // namespace mlir
