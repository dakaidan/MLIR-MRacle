#include "mlir-mr/passes/MetamorphicPass.h"
#include "mlir-mr/context/context.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <random>
#include <string>
#include <utility>

namespace mlir {

#define GEN_PASS_DECL_METAMORPHICPASS
#include "MetamorphicPass.inc"

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
//  - is a direct child of the thread body (moving it out of nested control
//    flow such as scf.if would drop its guard), and
//  - only uses operands defined outside the enclosing parallel region.
// Such RMWs can safely be moved between threads.
static bool isThreadAtomic(memref::AtomicRMWOp atomic) {
    omp::ParallelOp enclosing = atomic->getParentOfType<omp::ParallelOp>();
    if (!enclosing || !atomic.getResult().use_empty())
        return false;

    Operation *container = getThreadContainer(atomic);
    if (!container || atomic->getBlock() != &container->getRegion(0).front())
        return false;

    return llvm::all_of(atomic->getOperands(), [&](Value v) {
        Operation *defOp = v.getDefiningOp();
        return defOp && !enclosing->isAncestor(defOp);
    });
}

// True if two atomic RMWs are interchangeable: same value, memref and
// indices, so they touch exactly the same location with the same amount.
static bool operandsEqual(memref::AtomicRMWOp a, memref::AtomicRMWOp b) {
    if (a.getNumOperands() != b.getNumOperands())
        return false;
    for (auto [lhs, rhs] : llvm::zip_equal(a->getOperands(), b->getOperands()))
        if (lhs != rhs)
            return false;
    return true;
}

// True if the op executes inside a thread region (omp.section or omp.parallel).
static bool isOpInThread(Operation *op) {
    return op->getParentOfType<omp::SectionOp>() ||
           op->getParentOfType<omp::ParallelOp>();
}

// True if the memref is private to the executing thread: its allocation op
// sits inside an omp thread region and dominates the load (same region or an
// enclosing region), so no other thread can reach it through SSA. Allocations
// made in the enclosing function body (outside omp.parallel) are shared.
static bool isThreadLocalMemref(Value memref, Operation *use) {
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocaOp, memref::AllocOp>(defOp))
        return false;
    return isOpInThread(defOp) &&
           defOp->getParentRegion()->isAncestor(use->getParentRegion());
}

// checks that the given run of loads is eligible for reordering, i.e.:
//  - the run has at least two loads
//  - if inside a thread, only loads of thread-local memrefs are reorderable,
//    since shared memrefs can race with other threads
//  - no load in the run depends on the result of another load in the run
static bool checkLoadRun(SmallVector<memref::LoadOp> &run,
                         bool requireLocalMemrefs) {
    if (run.size() < 2)
        return false;
    if (requireLocalMemrefs)
        for (auto load : run)
            if (!isThreadLocalMemref(load.getMemref(), load))
                return false;
    for (size_t i = 0; i < run.size(); ++i)
        for (size_t j = i + 1; j < run.size(); ++j)
            if (llvm::is_contained(run[j]->getOperands(), run[i].getResult()) ||
                llvm::is_contained(run[i]->getOperands(), run[j].getResult()))
                return false;
    return true;
}

}
#define GEN_PASS_DEF_METAMORPHICPASS
#include "MetamorphicPass.inc"

struct MetamorphicPass
    : public impl::MetamorphicPassBase<MetamorphicPass> {

    using impl::MetamorphicPassBase<MetamorphicPass>::MetamorphicPassBase;

    mlir_mr::RunInfo *runInfo = nullptr;

    void runOnOperation() override {
        ModuleOp op = getOperation();
        IRRewriter rewriter(op->getContext());
        std::mt19937 rng(seed.getValue());

        using Transform = bool (MetamorphicPass::*)(func::FuncOp, RewriterBase &, std::mt19937 &);

        // map of transform names to member function pointers
        static const llvm::StringMap<Transform> kTransformMap = {
            {"multiset-permutation", &MetamorphicPass::tryMultisetPermutation},
            {"load-reordering", &MetamorphicPass::tryLoadReordering},
            {"insert-fence", &MetamorphicPass::tryInsertFence},
            // {"insert-thread", &MetamorphicPass::tryInsertThread},
            {"insert-atomic-rmw", &MetamorphicPass::tryInsertAtomicRMWInThread},
            {"local-store-duplication", &MetamorphicPass::tryLocalStoreDuplication},
            {"try-insert-random-arith", &MetamorphicPass::tryInsertRandomArith},
            {"try-insert-comparison", &MetamorphicPass::tryInsertComparison},
        };

        SmallVector<std::pair<std::string, Transform>, 4> transforms;
        for (auto &kv : kTransformMap)
            transforms.push_back({kv.getKey().str(), kv.second});

        std::string requested = transform.getValue();

        // parse the comma-separated requested list, empty means "pick at random"
        SmallVector<llvm::StringRef, 4> requestedNames;
        if (!requested.empty())
            llvm::StringRef(requested).split(requestedNames, ',');

        // validate every requested transform name before filtering
        for (llvm::StringRef name : requestedNames)
            if (!kTransformMap.contains(name)) {
                if (runInfo)
                    runInfo->error = "unknown transform '" + name.str() + "'";
                signalPassFailure();
                return;
            }

        // if specific transforms are requested, filter the list to only those
        // else keep the full list and pick at random
        if (!requestedNames.empty())
            llvm::erase_if(transforms, [&](const auto &t) {
                return !llvm::is_contained(requestedNames, t.first);
            });
        std::sort(transforms.begin(), transforms.end(),
                  [](const auto &a, const auto &b) {
                      return a.first < b.first;
                  });
        if (runInfo)
            for (llvm::StringRef name : requestedNames)
                runInfo->requestedTransforms.push_back(name.str());

        // collect all functions in the module
        SmallVector<func::FuncOp> funcs;
        op.walk([&](func::FuncOp f) { funcs.push_back(f); });
        if (funcs.empty())
            return;

        // pick one function at random to transform
        std::uniform_int_distribution<size_t> dist(0, funcs.size() - 1);
        func::FuncOp target = funcs[dist(rng)];

        int transformCounter = 0;

        // keep applying random transformations until max applications is reached
        while (transformCounter < maxApply) {
            std::shuffle(transforms.begin(), transforms.end(), rng);

            bool applied = false;
            for (auto &t : transforms)
                if ((this->*t.second)(target, rewriter, rng)) {
                    transformCounter++;
                    if (runInfo) {
                        runInfo->appliedTransforms.push_back(
                            {t.first, target.getName().str()});
                        runInfo->transformApplied = true;
                    }
                    applied = true;
                    break;
                }

            if (!applied)
                break;
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
        llvm::DenseMap<std::pair<Value, arith::AtomicRMWKind>, size_t> firstIdx;
        size_t idx = 0;
        op.walk([&](memref::AtomicRMWOp atomic) {
            if (isThreadAtomic(atomic)) {
                auto key =
                    std::make_pair(atomic.getMemref(), atomic.getKind());
                auto &group = groups[key];
                // Only structurally identical RMWs (same value, memref and
                // indices) are interchangeable, so only add equal ops.
                if (group.empty() || operandsEqual(group.front(), atomic)) {
                    group.push_back(atomic);
                    if (!firstIdx.count(key))
                        firstIdx[key] = idx;
                }
            }
            ++idx;
        });

        // if group size > 2 and the kind is commutative and associative, add to candidates
        SmallVector<std::pair<Value, arith::AtomicRMWKind>> candidates;
        for (const auto &kv : groups)
            if (kv.second.size() >= 2 && isCommutativeAssociative(kv.first.second))
                candidates.push_back(kv.first);
        std::sort(candidates.begin(), candidates.end(),
                  [&](const auto &a, const auto &b) {
                      return firstIdx.lookup(a) < firstIdx.lookup(b);
                  });

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
    // - runs inside threads only reorder loads of thread-local memrefs
    // - runs in the single-threaded body may reorder loads of any memref
    bool tryLoadReordering(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        ModuleOp module = op->getParentOfType<ModuleOp>();

        if (!module)
            return false;

        llvm::DenseSet<func::FuncOp> threadReachable;

        // We compute the set of functions reachable from thread regions in the module,
        // as these functions would thread-reachable, execute parallelly and thus have restrictions on load reordering.
        // Since its parallel too, only thread-local memrefs can be reordered, as shared memrefs can race
        bool changed = true;
        while (changed) {
            changed = false;
            module.walk([&](func::CallOp call) {
                func::FuncOp caller = call->getParentOfType<func::FuncOp>();
                func::FuncOp callee = dyn_cast_or_null<func::FuncOp>(
                    SymbolTable::lookupSymbolIn(module, call.getCallee()));
                if (!caller || !callee)
                    return;
                if ((isOpInThread(call) || threadReachable.contains(caller)) &&
                    threadReachable.insert(callee).second)
                    changed = true;
            });
        }
        bool funcInThread = threadReachable.contains(op);

        SmallVector<SmallVector<memref::LoadOp>> eligibleRuns;

        // function def: iterate over all regions in the function and collect eligible runs of loads
        std::function<void(Region &)> processRegion = [&](Region &region) {
            for (auto &block : region) {
                if (!block.empty()) {
                    // the whole block shares one thread context
                    bool inThread = funcInThread || isOpInThread(&block.front());
                    SmallVector<memref::LoadOp> currentRun;

                    for (auto &op : block) {
                        // if load, add
                        if (auto load = dyn_cast<memref::LoadOp>(op)) {
                            currentRun.push_back(load);
                        } else {
                            // else, no more loads in this run, so check if eligible
                            if (checkLoadRun(currentRun, inThread))
                                eligibleRuns.push_back({std::move(currentRun)});
                            // reset for next run
                            currentRun.clear();
                        }
                    }

                    // flush a trailing run at the end of the block
                    if (checkLoadRun(currentRun, inThread))
                        eligibleRuns.push_back({std::move(currentRun)});
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

    // TODO: subset relation
    // randomly insert an omp.flush fence at a random point in the function
    bool tryInsertFence(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<Operation *> allOps;
        
        op.getBody().walk([&](Operation *innerOp) {
            // omp.sections regions only allow omp.section ops and a terminator
            // never place a flush as a direct child of one.
            if (isa<omp::SectionsOp>(innerOp->getParentOp()))
                return;
            allOps.push_back(innerOp);
        });

        if (allOps.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, allOps.size() - 1);

        rewriter.setInsertionPoint(allOps[dist(rng)]);
        omp::FlushOp::create(rewriter, op.getLoc(), ValueRange{});
        return true;
    }

//    bool tryInsertThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
//
//        // Walk to find an existing omp.sections op
//        SmallVector<omp::SectionsOp> sectionsOps;
//        op.walk([&](omp::SectionsOp s) {
//            sectionsOps.push_back(s);
//        });
//
//        if (sectionsOps.empty())
//            return false;
//        
//        std::uniform_int_distribution<size_t> dist(0, sectionsOps.size() - 1);
//        auto &section = sectionsOps[dist(rng)];
//
//        // if no omp.sections op found, return false and do not apply
//        if (!section)
//            return false;
//
//        Region &region = section.getRegion();
//        if (region.empty())
//            return false;
//
//        Block &sectionBlock = region.front();
//        if (sectionBlock.empty() || !sectionBlock.back().hasTrait<OpTrait::IsTerminator>())
//            return false;
//
//        // Insert a new section before the sections terminator.
//        Operation *terminator = sectionBlock.getTerminator();
//        rewriter.setInsertionPoint(terminator);
//
//        auto newSection = omp::SectionOp::create(rewriter, op.getLoc());
//
//        // Create a new block for the new section
//        Block *body = rewriter.createBlock(&newSection.getRegion());
//        for (Type ty : sectionBlock.getArgumentTypes())
//            body->addArgument(ty, op.getLoc());
//        omp::TerminatorOp::create(rewriter, op.getLoc());
//
//        return true;
//    }

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

    // Randomly insert a fresh memref and a chain of arith ops
    // Keep going until max chain length of 5 is hit, or a store is inserted back to the memref, which stops the chain
    bool tryInsertRandomArith(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<Operation *> allOps;
        
        op.getBody().walk([&](Operation *innerOp) {
            // omp.sections regions only allow omp.section ops and a terminator,
            // so never insert generated ops as a direct child of one.
            if (isa<omp::SectionsOp>(innerOp->getParentOp()))
                return;
            allOps.push_back(innerOp);
        });

        if (allOps.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, allOps.size() - 1);
        rewriter.setInsertionPoint(allOps[dist(rng)]);

        Location loc = op.getLoc();

        // i64 1-D memref
        auto memrefType = MemRefType::get(
            {static_cast<int64_t>(std::uniform_int_distribution<int>(1, 16)(rng))},
            rewriter.getI64Type());
        auto memref = memref::AllocaOp::create(rewriter, loc, memrefType,
                                               /*dynamicSizes=*/ValueRange{},
                                               /*alignment=*/IntegerAttr{});

        auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value acc = memref::LoadOp::create(rewriter, loc, memref, ValueRange{zeroIdx});

        std::uniform_int_distribution<int64_t> intDist(-100, 100);
        auto makeConst = [&]() {
            return arith::ConstantOp::create(
                rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(intDist(rng)));
        };

        std::uniform_int_distribution<int> opDist(0, 2);
        auto makeArith = [&](Value lhs) -> Value {
            Value rhs = makeConst();
            switch (opDist(rng)) {
            case 0: return arith::AddIOp::create(rewriter, loc, lhs, rhs);
            case 1: return arith::SubIOp::create(rewriter, loc, lhs, rhs);
            default: return arith::MulIOp::create(rewriter, loc, lhs, rhs);
            }
        };

        // random number of chained arith ops, capped at 5
        int chain = std::uniform_int_distribution<int>(1, 5)(rng);
        for (int i = 0; i < chain; ++i)
            acc = makeArith(acc);

        // coinflip between continuing the chain or storing back to the memref
        std::bernoulli_distribution coin(0.5);
        while (coin(rng)) {
            if (chain < 5 && coin(rng)) {
                acc = makeArith(acc);
                ++chain;
            } else {
                memref::StoreOp::create(rewriter, loc, acc, memref, ValueRange{zeroIdx});
                break;
            }
        }

        return true;
    }

    bool tryInsertComparison(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        // Collect integer-typed results defined inside thread regions
        // (omp.section / omp.parallel). Comparing a value with itself (x == x)
        // is well-defined for any integer, so no UB regardless of the value.
        SmallVector<Value> candidates;
        op.getBody().walk([&](Operation *innerOp) {
            if (!isOpInThread(innerOp) ||
                innerOp->hasTrait<OpTrait::IsTerminator>())
                return;
            for (Value res : innerOp->getResults())
                if (res.getType().isIntOrIndex()) {
                    candidates.push_back(res);
                    return;
                }
        });

        if (candidates.empty())
            return false;

        // pick a random available result
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        Value value = candidates[dist(rng)];

        // insert 1-4 comparisons right after the defining op so the result is
        // in scope and the comparison runs in the same thread
        int numComparisons = std::uniform_int_distribution<int>(1, 4)(rng);
        rewriter.setInsertionPointAfter(value.getDefiningOp());

        Location loc = op.getLoc();
        for (int i = 0; i < numComparisons; ++i)
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                  value, value);

        return true;
    }
};

std::unique_ptr<Pass> createMetamorphicPass(
    int seed, mlir_mr::RunInfo *runInfo, std::string transform,
    int maxApply) {
    MetamorphicPassOptions options;
    options.seed = seed;
    options.transform = transform;
    options.maxApply = maxApply;
    auto pass = std::make_unique<MetamorphicPass>(options);
    pass->runInfo = runInfo;
    return pass;
}

} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "MetamorphicPass.inc"
#include <list>
#include <set>
#include <map>
#include <cmath>
#include <iterator>
