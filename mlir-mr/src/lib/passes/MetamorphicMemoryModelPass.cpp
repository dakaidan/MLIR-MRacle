#include "mlir-mr/passes/MetamorphicMemoryModelPass.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
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
        ModuleOp op = getOperation();
        IRRewriter rewriter(op->getContext());
        std::mt19937 rng(seed.getValue());

        using Transform = bool (MetamorphicMemoryModelPass::*)(func::FuncOp, RewriterBase &, std::mt19937 &);

        // map of transform names to member function pointers
        static const llvm::StringMap<Transform> kTransformMap = {
            {"multiset-permutation", &MetamorphicMemoryModelPass::tryMultisetPermutation},
            {"load-reordering", &MetamorphicMemoryModelPass::tryLoadReordering},
            {"insert-fence", &MetamorphicMemoryModelPass::tryInsertFence},
            {"insert-thread", &MetamorphicMemoryModelPass::tryInsertThread},
            {"insert-atomic-rmw", &MetamorphicMemoryModelPass::tryInsertAtomicRMWInThread},
            {"local-store-duplication", &MetamorphicMemoryModelPass::tryLocalStoreDuplication}
        };

        // pass option, from flag
        // keep the name alongside each transform so debug output can report which one was picked
        SmallVector<std::pair<std::string, Transform>, 4> transforms;

        // if empty use all transforms, otherwise use the ones specified in the flag
        if (this->transforms.empty()) {
            for (auto &kv : kTransformMap)
                transforms.push_back({kv.getKey().str(), kv.second});
        } else {
            for (const auto &name : this->transforms) {
                auto it = kTransformMap.find(name);
                if (it != kTransformMap.end())
                    transforms.push_back({it->getKey().str(), it->second});
            }
            if (transforms.empty())
                return;
        }

        // collect all functions in the module
        SmallVector<func::FuncOp> funcs;
        op.walk([&](func::FuncOp f) { funcs.push_back(f); });
        if (funcs.empty())
            return;

        // pick one function at random to transform
        std::uniform_int_distribution<size_t> dist(0, funcs.size() - 1);
        func::FuncOp target = funcs[dist(rng)];

        // shuffle the order of the passes to apply
        std::shuffle(transforms.begin(), transforms.end(), rng);

        // keep applying passes until one succeeds
        for (auto &t : transforms)
            if ((this->*t.second)(target, rewriter, rng)) {
                if (debug.getValue())
                    llvm::errs() << "mlir-mr applied transformation '"
                                 << t.first << "' in function '"
                                 << target.getName() << "'\n";
                return;
            }
    }

private:
    // TODO: for MRs, generalise to all functions with FunctionOpInterface, not just func::FuncOp

    // TODO: generalise for more atomic operations?
    bool tryMultisetPermutation(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

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

    // randomly reorder eligible runs of loads in the function
    // - assumes loads are independent and do not depend on any stores to the same memref
    // - 
    bool tryLoadReordering(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

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

        // randomly select an eligible run
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

    // randomly insert an omp.flush fence at a random point in the function
    bool tryInsertFence(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<Operation *> allOps;
        
        op.getBody().walk([&](Operation *innerOp) {
            allOps.push_back(innerOp);
        });

        if (allOps.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, allOps.size() - 1);

        rewriter.setInsertionPoint(allOps[dist(rng)]);
        omp::FlushOp::create(rewriter, op.getLoc(), ValueRange{});
        return true;
    }

    bool tryInsertThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // Walk to find an existing omp.sections op
        SmallVector<omp::SectionsOp> sectionsOps;
        op.walk([&](omp::SectionsOp s) {
            sectionsOps.push_back(s);
        });

        if (sectionsOps.empty())
            return false;
        
        std::uniform_int_distribution<size_t> dist(0, sectionsOps.size() - 1);
        auto &section = sectionsOps[dist(rng)];

        // if no omp.sections op found, return false and do not apply
        if (!section)
            return false;

        Region &region = section.getRegion();
        if (region.empty())
            return false;

        Block &sectionBlock = region.front();
        if (sectionBlock.empty() || !sectionBlock.back().hasTrait<OpTrait::IsTerminator>())
            return false;

        // Insert a new section before the sections terminator.
        Operation *terminator = sectionBlock.getTerminator();
        rewriter.setInsertionPoint(terminator);

        auto newSection = omp::SectionOp::create(rewriter, op.getLoc());

        // Create a new block for the new section
        Block *body = rewriter.createBlock(&newSection.getRegion());
        for (Type ty : sectionBlock.getArgumentTypes())
            body->addArgument(ty, op.getLoc());
        omp::TerminatorOp::create(rewriter, op.getLoc());

        return true;
    }

    bool tryInsertAtomicRMWInThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<omp::SectionOp> candidates;

        // Walk to find all omp.section ops in the function body
        op.getBody().walk([&](Operation *innerOp) {
            if (auto sectionOp = dyn_cast<omp::SectionOp>(innerOp)) {
                candidates.push_back(sectionOp);
            }
        });

        if (candidates.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        omp::SectionOp section = candidates[dist(rng)];

        if (section.getRegion().empty())
            return false;

        // Insert into the section body, before its terminator.
        rewriter.setInsertionPoint(section.getRegion().front().getTerminator());

        Location loc = section.getLoc();

        // Dummy 0-d memref on this thread's stack and an atomic RMW that adds 0
        // to it: a no-op that still exercises the atomic memory path.
        auto memrefType = MemRefType::get({}, rewriter.getI32Type());
        auto dummyMemref = memref::AllocaOp::create(rewriter, loc, memrefType,
                                                    /*dynamicSizes=*/ValueRange{},
                                                    /*alignment=*/IntegerAttr{});
        auto dummyValue = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        memref::AtomicRMWOp::create(rewriter, loc, rewriter.getI32Type(),
                                    arith::AtomicRMWKind::addi, dummyValue,
                                    dummyMemref, ValueRange{});

        return true;
    }

    bool tryLocalStoreDuplication(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // the sections that each memref is stored to, so we can find memrefs that are only stored to in one section
        llvm::DenseMap<Value, llvm::DenseSet<omp::SectionOp>> memrefToSections;

        // get all memrefs that are stored to in omp.section ops and map them to the sections they are stored in
        op.getBody().walk([&](Operation *innerOp) {
            Value memref;
            if (auto load = dyn_cast<memref::LoadOp>(innerOp))
                memref = load.getMemref();
            else if (auto store = dyn_cast<memref::StoreOp>(innerOp))
                memref = store.getMemref();
            else if (auto atomic = dyn_cast<memref::AtomicRMWOp>(innerOp))
                memref = atomic.getMemref();
            else
                return;

            if (auto section = innerOp->getParentOfType<omp::SectionOp>())
                memrefToSections[memref].insert(section);
        });

        SmallVector<memref::StoreOp> candidates;

        // find all store ops that are in a section and whose memref is only stored to in that section
        op.getBody().walk([&](Operation *innerOp) {
            auto storeOp = dyn_cast<memref::StoreOp>(innerOp);
            if (!storeOp)
                return;

            auto section = storeOp->getParentOfType<omp::SectionOp>();
            if (!section)
                return;

            Value memref = storeOp.getMemref();

            // get the memref
            auto it = memrefToSections.find(memref);

            // if the memref is only stored to in this section, add the store op to candidates
            if (it != memrefToSections.end() && it->second.size() == 1 &&
                it->second.contains(section))
                candidates.push_back(storeOp);
        });

        if (candidates.empty())
            return false;

        // pick one store op at random and duplicate it after itself
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        memref::StoreOp store = candidates[dist(rng)];

        // Insert a duplicate store after the original store
        rewriter.setInsertionPointAfter(store);
        rewriter.clone(*store);

        return true;
    }
};

std::unique_ptr<Pass> createMetamorphicMemoryModelPass(int seed, const std::string &transforms, bool debug) {
    MetamorphicMemoryModelPassOptions options;
    options.seed = seed;
    options.debug = debug;
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
#include <set>
