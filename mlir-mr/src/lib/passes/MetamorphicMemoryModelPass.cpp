#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <random>
#include <utility>

namespace mlir {

#define GEN_PASS_DECL_METAMORPHICMEMORYMODELPASS
#include "MetamorphicMemoryModelPass.inc"

// Helper functions should go here in anonymous namespace (does not show up in global namespace)
namespace {

// True if the given AtomicRMWKind is commutative and associative.
static bool isCommutativeAssociative(arith::AtomicRMWKind kind) {
    switch (kind) {
    case arith::AtomicRMWKind::addi:
    case arith::AtomicRMWKind::andi:
    case arith::AtomicRMWKind::ori:
    case arith::AtomicRMWKind::xori:
    case arith::AtomicRMWKind::maxs:
    case arith::AtomicRMWKind::maxu:
    case arith::AtomicRMWKind::mins:
    case arith::AtomicRMWKind::minu:
        return true;
    default:
        return false;
    }
}

// Get the innermost omp.section that contains the atomic RMW, or the
// enclosing omp.parallel. Returns nullptr if the RMW is not inside a thread.
static Operation *getThreadContainer(memref::AtomicRMWOp atomic) {
    if (auto section = atomic->getParentOfType<omp::SectionOp>())
        return section;
    return atomic->getParentOfType<omp::ParallelOp>();
}

// A thread‑atomic RMW:
//  - runs inside an omp region,
//  - has an unused result (store‑only semantic),
//  - only uses operands defined outside the enclosing parallel region.
// Such RMWs can safely be moved between threads.
static bool isThreadAtomic(memref::AtomicRMWOp atomic) {
    omp::ParallelOp enclosing = atomic->getParentOfType<omp::ParallelOp>();
    if (!enclosing || !atomic.getResult().use_empty())
        return false;

    return llvm::all_of(atomic->getOperands(), [&](Value v) {
        Operation *defOp = v.getDefiningOp();
        return defOp && !enclosing->isAncestor(defOp);
    });
}

// checks that the given run of loads is eligible for reordering, i.e.:
//  - the run has at least two loads
//  - none of the loads read from a memref that has been stored to
//  - no load in the run depends on the result of another load in the run
static bool checkLoadRun(SmallVector<memref::LoadOp> &run,
                         const llvm::DenseSet<Value> &storedMemrefs) {
    if (run.size() < 2)
        return false;
    for (auto load : run)
        if (storedMemrefs.contains(load.getMemref()))
            return false;
    for (size_t i = 0; i < run.size(); ++i)
        for (size_t j = i + 1; j < run.size(); ++j)
            if (llvm::is_contained(run[j]->getOperands(), run[i].getResult()) ||
                llvm::is_contained(run[i]->getOperands(), run[j].getResult()))
                return false;
    return true;
}

}
#define GEN_PASS_DEF_METAMORPHICMEMORYMODELPASS
#include "MetamorphicMemoryModelPass.inc"

struct MetamorphicMemoryModelPass
    : public impl::MetamorphicMemoryModelPassBase<MetamorphicMemoryModelPass> {

    using impl::MetamorphicMemoryModelPassBase<MetamorphicMemoryModelPass>::MetamorphicMemoryModelPassBase;

    void runOnOperation() override {
        func::FuncOp op = getOperation();
        IRRewriter rewriter(op->getContext());
        std::mt19937 rng(seed.getValue());

        using Transform = bool (MetamorphicMemoryModelPass::*)(func::FuncOp, RewriterBase &, std::mt19937 &);

        // Register possible passes
        SmallVector<Transform> transforms = {
            &MetamorphicMemoryModelPass::tryApplyMultisetPermutation,
            &MetamorphicMemoryModelPass::tryApplyLoadReordering,
        };

        // shuffle the order of the passes to apply
        std::shuffle(transforms.begin(), transforms.end(), rng);

        // keep applying passes until one succeeds
        for (auto &t : transforms)
            if ((this->*t)(op, rewriter, rng))
                return;
    }

private:
    bool tryApplyMultisetPermutation(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // Groups of thread RMW ops keyed by (memref, AtomicRMWKind).
        using RmwGroupMap =
            llvm::DenseMap<std::pair<Value, arith::AtomicRMWKind>,
                   SmallVector<memref::AtomicRMWOp>>;

        RmwGroupMap groups;
        op.walk([&](memref::AtomicRMWOp atomic) {
            if (isThreadAtomic(atomic))
                groups[{atomic.getMemref(), atomic.getKind()}].push_back(atomic);
        });

        SmallVector<std::pair<Value, arith::AtomicRMWKind>> candidates;
        for (const auto &kv : groups)
            if (kv.second.size() >= 2 && isCommutativeAssociative(kv.first.second))
                candidates.push_back(kv.first);

        if (candidates.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        SmallVector<memref::AtomicRMWOp> atomics = groups[candidates[dist(rng)]];

        SmallVector<Operation *> containers;
        containers.reserve(atomics.size());
        for (memref::AtomicRMWOp a : atomics)
        containers.push_back(getThreadContainer(a));

        std::shuffle(atomics.begin(), atomics.end(), rng);

        for (size_t i = 0; i < atomics.size(); ++i) {
            Operation *anchor = containers[i]->getRegion(0).front().getTerminator();
            rewriter.moveOpBefore(atomics[i], anchor);
        }
        return true;
    }

    bool tryApplyLoadReordering(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // firstly, collect all memrefs and that have been stored to in the function
        // to avoid reordering loads that depend on a store to the same memref
        // thus avoiding race conditions
        llvm::DenseSet<Value> storedMemrefs;
        op.walk([&](memref::StoreOp store) {
            storedMemrefs.insert(store.getMemref());
        });
        op.walk([&](memref::AtomicRMWOp atomic) {
            storedMemrefs.insert(atomic.getMemref());
        });

        SmallVector<SmallVector<memref::LoadOp>> eligibleRuns;

        // function def: iterate over all regions in the function and collect eligible runs of loads
        std::function<void(Region &)> processRegion = [&](Region &region) {
            for (auto &block : region) {
                SmallVector<memref::LoadOp> currentRun;

                for (auto &op : block) {
                    // if load, add
                    if (auto load = dyn_cast<memref::LoadOp>(op)) {
                        currentRun.push_back(load);
                    } else {
                        // else, no more loads in this run, so check if eligible
                        if (checkLoadRun(currentRun, storedMemrefs))
                            eligibleRuns.push_back({std::move(currentRun)});
                        // reset for next run
                        currentRun.clear();
                    }
                }

                // iterate over all regions in the block and process them recursively
                for (auto &op : block)
                    for (auto &r : op.getRegions())
                        processRegion(r);
            }
        };

        // actual function call, process regions in the function
        processRegion(op.getRegion());

        if (eligibleRuns.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, eligibleRuns.size() - 1);
        auto &run = eligibleRuns[dist(rng)];

        SmallVector<memref::LoadOp> shuffled = run;
        
        // keep shuffling until the order actually differs from the original run
        do {
            std::shuffle(shuffled.begin(), shuffled.end(), rng);
        } while (shuffled == run);

        Operation *firstPos = run[0];

        // rechain in place pattern
        for (size_t i = 0; i < shuffled.size(); ++i) {
            if (i == 0) {
                if (shuffled[i] != firstPos)
                    rewriter.moveOpBefore(shuffled[i], firstPos);
            } else {
                rewriter.moveOpAfter(shuffled[i], shuffled[i - 1]);
            }
        }

        return true;
    }
};

std::unique_ptr<Pass> createMetamorphicMemoryModelPass() {
    return std::make_unique<MetamorphicMemoryModelPass>();
}

} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "MetamorphicMemoryModelPass.inc"#include <iterator>
