#include "mlir-mr/passes/MetamorphicMemoryModelPass.h"

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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <random>
#include <string>
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

        // map of transform names to member function pointers
        static const llvm::StringMap<Transform> kTransformMap = {
            {"multiset-permutation", &MetamorphicMemoryModelPass::tryApplyMultisetPermutation},
            {"load-reordering", &MetamorphicMemoryModelPass::tryApplyLoadReordering},
            {"insert-fence", &MetamorphicMemoryModelPass::tryInsertFenceRandom},
        };

        // pass option, from flag
        SmallVector<Transform> transforms;

        // if empty use all transforms, otherwise use the ones specified in the flag
        if (this->transforms.empty()) {
            for (auto &kv : kTransformMap)
                transforms.push_back(kv.second);
        } else {
            for (const auto &name : this->transforms) {
                auto it = kTransformMap.find(name);
                if (it != kTransformMap.end())
                    transforms.push_back(it->second);
            }
            if (transforms.empty())
                return;
        }

        // shuffle the order of the passes to apply
        std::shuffle(transforms.begin(), transforms.end(), rng);

        // keep applying passes until one succeeds
        for (auto &t : transforms)
            if ((this->*t)(op, rewriter, rng))
                return;
    }

private:

    // TODO: generalise for more atomic operations?
    bool tryApplyMultisetPermutation(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // Groups of thread RMW ops keyed by (memref, AtomicRMWKind).
        using RmwGroupMap =
            llvm::DenseMap<std::pair<Value, arith::AtomicRMWKind>,
                   SmallVector<memref::AtomicRMWOp>>;

        // find all thread-atomic RMWs and group them by (memref, kind)
        RmwGroupMap groups;
        op.walk([&](memref::AtomicRMWOp atomic) {
            if (isThreadAtomic(atomic))
                groups[{atomic.getMemref(), atomic.getKind()}].push_back(atomic);
        });

        // if group size > 2 and the kind is commutative and associative, add to candidates
        SmallVector<std::pair<Value, arith::AtomicRMWKind>> candidates;
        for (const auto &kv : groups)
            if (kv.second.size() >= 2 && isCommutativeAssociative(kv.first.second))
                candidates.push_back(kv.first);

        if (candidates.empty())
            return false;

        // randomly select a candidate group and shuffle the RMWs within that group
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        SmallVector<memref::AtomicRMWOp> atomics = groups[candidates[dist(rng)]];

        SmallVector<Operation *> containers;
        containers.reserve(atomics.size());

        // for each atomc, store the innermost omp.section or omp.parallel that contains it
        for (memref::AtomicRMWOp a : atomics)
            containers.push_back(getThreadContainer(a));

        // randomise the thread containers
        std::uniform_int_distribution<size_t> containerDist(0, containers.size() - 1);

        // for each RMW, move it to the end of a randomly chosen thread container
        for (memref::AtomicRMWOp atomic : atomics) {
            Operation *container = containers[containerDist(rng)];
            Operation *anchor = container->getRegion(0).front().getTerminator();
            rewriter.moveOpBefore(atomic, anchor);
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

    bool tryInsertFenceRandom(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<Operation *> allOps;
        op.walk([&](Operation *op) {
            allOps.push_back(op);
        });

        if (allOps.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, allOps.size() - 1);

        rewriter.setInsertionPoint(allOps[dist(rng)]);
        rewriter.create<omp::FlushOp>(op.getLoc(), ValueRange{});
        return true;
    }
};

std::unique_ptr<Pass> createMetamorphicMemoryModelPass(int seed, const std::string &transforms) {
    MetamorphicMemoryModelPassOptions options;
    options.seed = seed;
    if (!transforms.empty()) {
        SmallVector<StringRef, 4> names;
        StringRef(transforms).split(names, ',', -1, false);
        for (auto name : names)
            options.transforms.push_back(name.trim().str());
    }
    return std::make_unique<MetamorphicMemoryModelPass>(options);
}

} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "MetamorphicMemoryModelPass.inc"
#include <map>
#include <iterator>
