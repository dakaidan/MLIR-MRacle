#include "mlir-mracle/passes/MetamorphicPass.h"
#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "mlir-mracle/context/context.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <random>
#include <string>
#include <utility>

namespace mlir {

#define GEN_PASS_DECL_METAMORPHICPASS
#include "MetamorphicPass.inc"

#define GEN_PASS_DEF_METAMORPHICPASS
#include "MetamorphicPass.inc"

struct MetamorphicPass
    : public impl::MetamorphicPassBase<MetamorphicPass> {

    using impl::MetamorphicPassBase<MetamorphicPass>::MetamorphicPassBase;

    mlir_mracle::RunInfo *runInfo = nullptr;

    void runOnOperation() override {
        ModuleOp op = getOperation();
        IRRewriter rewriter(op->getContext());
        std::mt19937 rng(seed.getValue());

        // Resolve the registry for the requested memory model. Generic
        // transforms are always included; an empty model adds no
        // memory-model-specific ones.
        auto registry = metamorphic::getTransforms(model.getValue());

        auto findSpec =
            [&](llvm::StringRef name) -> const metamorphic::MetamorphicTransform * {
            for (const auto *spec : registry)
                if (spec->getName() == name)
                    return spec;
            return nullptr;
        };

        std::string requested = transform.getValue();

        // parse the comma-separated requested list, empty means "pick at random"
        SmallVector<llvm::StringRef, 4> requestedNames;
        if (!requested.empty())
            llvm::StringRef(requested).split(requestedNames, ',');

        // validate every requested transform name before filtering
        for (llvm::StringRef name : requestedNames)
            if (!findSpec(name)) {
                if (runInfo)
                    runInfo->error = "unknown transform '" + name.str() + "'";
                signalPassFailure();
                return;
            }

        // if specific transforms are requested, filter the list to only those
        // else keep the full list and pick at random
        SmallVector<const metamorphic::MetamorphicTransform *> transforms;
        for (const auto *spec : registry)
            if (requestedNames.empty() ||
                llvm::is_contained(requestedNames, spec->getName()))
                transforms.push_back(spec);
        std::sort(transforms.begin(), transforms.end(),
                  [](const metamorphic::MetamorphicTransform *a,
                     const metamorphic::MetamorphicTransform *b) {
                      return a->getName() < b->getName();
                  });
        if (runInfo)
            for (llvm::StringRef name : requestedNames)
                runInfo->requestedTransforms.push_back(name.str());

        // collect all functions in the module
        SmallVector<func::FuncOp> funcs;
        op.walk([&](func::FuncOp f) { funcs.push_back(f); });
        if (funcs.empty())
            return;

        // Shuffle the functions so each one gets its own random tries.
        // Every function receives up to kMaxTransformAttempts independent
        // random attempts; a function that fails all of them is abandoned in
        // favour of the next one, and only when every function has been
        // exhausted do we signal failure.
        std::shuffle(funcs.begin(), funcs.end(), rng);
        constexpr int kMaxTransformAttempts = 3;
        for (func::FuncOp target : funcs) {
            for (int attempt = 0; attempt < kMaxTransformAttempts; ++attempt) {
                int transformCounter = 0;
                std::optional<mlir_mracle::OutcomeRelation> aggregate;

                // keep applying random transformations until max applications
                // is reached; transforms whose relation would contradict the
                // direction already established are excluded from the draw
                while (transformCounter < maxApply) {
                    SmallVector<const metamorphic::MetamorphicTransform *>
                        allowed;
                    for (const metamorphic::MetamorphicTransform *spec :
                         transforms)
                        if (metamorphic::canApplyAfter(
                                aggregate.value_or(
                                    mlir_mracle::OutcomeRelation::Equality),
                                spec->getRelation()))
                            allowed.push_back(spec);
                    if (allowed.empty())
                        break;

                    std::shuffle(allowed.begin(), allowed.end(), rng);
                    bool applied = false;
                    for (const metamorphic::MetamorphicTransform *spec :
                         allowed)
                        if (spec->apply(target, rewriter, rng)) {
                            transformCounter++;
                            aggregate =
                                aggregate
                                    ? metamorphic::composeRelation(
                                          *aggregate, spec->getRelation())
                                    : spec->getRelation();
                            if (runInfo) {
                                runInfo->appliedTransforms.push_back(
                                    {spec->getName().str(),
                                     target.getName().str()});
                                runInfo->transformApplied = true;
                            }
                            llvm::errs() << "=== AFTER "
                                         << spec->getName() << " ===\n";
                            target.print(llvm::errs());
                            llvm::errs() << "\n";
                            applied = true;
                            break;
                        }

                    if (!applied)
                        break;
                }

                if (aggregate) {
                    if (runInfo)
                        runInfo->relation = *aggregate;
                    return;
                }
            }
        }

        if (runInfo)
            runInfo->error = "tried applying transformations to every function " +
                             std::to_string(kMaxTransformAttempts) + " times";
        signalPassFailure();
    }
};

std::unique_ptr<Pass> createMetamorphicPass(
    int seed, mlir_mracle::RunInfo *runInfo, std::string transform,
    int maxApply, std::string model) {
    MetamorphicPassOptions options;
    options.seed = seed;
    options.transform = transform;
    options.maxApply = maxApply;
    options.model = model;
    auto pass = std::make_unique<MetamorphicPass>(options);
    pass->runInfo = runInfo;
    return pass;
}

} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "MetamorphicPass.inc"