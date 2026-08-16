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
#include "llvm/Support/ErrorHandling.h"
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

// A candidate insertion position for a generated op: insert before `op`, or
// append to `block` when `op` is null (empty / unterminated block).
struct InsertPoint {
    Operation *op = nullptr;
    Block *block = nullptr;
};

// Collect every legal insertion position in the region. Blocks that are direct
// children of omp.sections regions are excluded, since those regions only
// allow omp.section ops and a terminator. Empty blocks and blocks that do not
// end in a terminator are included, so generated ops can also land in empty
// bodies.
static void collectInsertPoints(Region &region,
                                SmallVectorImpl<InsertPoint> &points) {
    for (Block &block : region) {
        if (!isa<omp::SectionsOp>(block.getParentOp())) {
            for (Operation &op : block)
                points.push_back({&op, nullptr});
            if (block.empty() ||
                !block.back().hasTrait<OpTrait::IsTerminator>())
                points.push_back({nullptr, &block});
        }
        for (Operation &op : block)
            for (Region &child : op.getRegions())
                collectInsertPoints(child, points);
    }
}

// Collect maximal runs of consecutive ops in each block that satisfy `pred`.
// A run is flushed whenever an op fails the predicate or a block ends, so
// every returned run is a contiguous slice of ops within a single block.
template <typename PredT>
static SmallVector<SmallVector<Operation *>>
collectOpRuns(Operation *root, PredT pred) {
    SmallVector<SmallVector<Operation *>> runs;
    root->walk([&](Block *block) {
        SmallVector<Operation *> run;
        auto flush = [&]() {
            if (!run.empty()) {
                runs.push_back(std::move(run));
                run.clear();
            }
        };
        for (Operation &op : *block) {
            if (pred(&op))
                run.push_back(&op);
            else
                flush();
        }
        flush();
    });
    return runs;
}

// An op is wrappable if it is not a terminator, not an omp.section (which
// must stay inside its omp.sections region), and either has no results or
// only results used inside its own regions, so moving it into a nested
// region cannot break SSA.
static bool isWrappable(Operation *innerOp) {
    if (innerOp->hasTrait<OpTrait::IsTerminator>() ||
        isa<omp::SectionOp>(innerOp))
        return false;
    if (innerOp->getNumResults() == 0)
        return true;
    for (Value res : innerOp->getResults())
        for (Operation *user : res.getUsers())
            if (!innerOp->isAncestor(user))
                return false;
    return true;
}

// A runtime i1 condition that canonicalization cannot fold: an integer-typed
// SSA value already in scope (function argument, or a non-constant op result
// before the anchor) compared against zero. Returns null if none exists.
static Value makeRuntimeCondition(Operation *anchor, RewriterBase &rewriter,
                                  Location loc, std::mt19937 &rng) {
    SmallVector<Value> candidates;
    if (auto func = anchor->getParentOfType<func::FuncOp>())
        for (BlockArgument arg : func.getBody().getArguments())
            if (arg.getType().isSignlessInteger() ||
                arg.getType().isIndex())
                candidates.push_back(arg);
    if (Block *block = anchor->getBlock())
        for (Operation &op : block->getOperations()) {
            if (&op == anchor)
                break;
            if (isa<arith::ConstantOp>(op))
                continue;
            for (Value res : op.getResults())
                if (res.getType().isSignlessInteger() ||
                    res.getType().isIndex())
                    candidates.push_back(res);
        }
    if (candidates.empty())
        return nullptr;
    Value v =
        candidates[std::uniform_int_distribution<size_t>(0,
                                                         candidates.size() - 1)(rng)];
    if (v.getType().isInteger(1))
        return v;
    Value zero;
    if (v.getType().isIndex())
        zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    else
        zero = arith::ConstantOp::create(
            rewriter, loc, v.getType(), IntegerAttr::get(v.getType(), 0));
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, v,
                                 zero);
}

// Generates a self-contained sequence of ops at the rewriter's current
// insertion point: a fresh alloca, its initialising store, a load, and a chain
// of arith and/or memory ops that may store back to the alloca. Everything is
// thread-local, so the sequence is safe to place inside any thread region.
class ArithGenerator {
public:
    ArithGenerator(std::mt19937 &rng) : rng(rng) {}

    // Which step kinds a generated chain may contain.
    enum class ChainMode { Arith, Memref, Mixed };

    // arith-only chain (addi/subi/muli ops)
    void generate(RewriterBase &rewriter, Location loc) {
        auto [memref, zeroIdx] = makeMemrefAlloc(rewriter, loc);
        generateChain(rewriter, loc, memref, zeroIdx, ChainMode::Arith);
    }

    // chain of loads/stores to the memref only
    void generateMemref(RewriterBase &rewriter, Location loc) {
        auto [memref, zeroIdx] = makeMemrefAlloc(rewriter, loc);
        generateChain(rewriter, loc, memref, zeroIdx, ChainMode::Memref);
    }

    // chain mixing arith ops and memory ops
    void generateMixed(RewriterBase &rewriter, Location loc) {
        auto [memref, zeroIdx] = makeMemrefAlloc(rewriter, loc);
        generateChain(rewriter, loc, memref, zeroIdx, ChainMode::Mixed);
    }

private:
    // Fresh thread-local alloca, initialised to 0 so loads never read
    // uninitialised memory. Returns it with its zero index.
    std::pair<Value, Value> makeMemrefAlloc(RewriterBase &rewriter, Location loc) {
        auto memrefType = MemRefType::get(
            {static_cast<int64_t>(std::uniform_int_distribution<int>(1, 16)(rng))},
            rewriter.getI64Type());
        Value memref = memref::AllocaOp::create(rewriter, loc, memrefType,
                                                /*dynamicSizes=*/ValueRange{},
                                                /*alignment=*/IntegerAttr{});
        Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto init = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(0));
        memref::StoreOp::create(rewriter, loc, init, memref,
                                ValueRange{zeroIdx});
        return {memref, zeroIdx};
    }

    // The general chain: a load, a run of steps of the given mode, then a
    // coinflip that either extends the chain (capped at 5) or stores back.
    void generateChain(RewriterBase &rewriter, Location loc, Value memref,
                       Value idx, ChainMode mode) {
        Value acc = memref::LoadOp::create(rewriter, loc, memref,
                                           ValueRange{idx});

        // random number of chained steps, capped at 5
        int chain = std::uniform_int_distribution<int>(1, 5)(rng);
        for (int i = 0; i < chain; ++i)
            acc = makeStep(rewriter, loc, memref, idx, acc, mode);

        // coinflip between continuing the chain or storing back to the memref
        std::bernoulli_distribution coin(0.5);
        while (coin(rng)) {
            if (chain < 5 && coin(rng)) {
                acc = makeStep(rewriter, loc, memref, idx, acc, mode);
                ++chain;
            } else {
                memref::StoreOp::create(rewriter, loc, acc, memref,
                                        ValueRange{idx});
                break;
            }
        }
    }

    // One step of the chain: an arith op, a memory op, or a random mix.
    Value makeStep(RewriterBase &rewriter, Location loc, Value memref,
                   Value idx, Value acc, ChainMode mode) {
        switch (mode) {
        case ChainMode::Arith:
            return makeArith(rewriter, loc, acc);
        case ChainMode::Memref:
            return makeMemref(rewriter, loc, memref, idx, acc);
        case ChainMode::Mixed:
            return std::bernoulli_distribution(0.5)(rng)
                       ? makeArith(rewriter, loc, acc)
                       : makeMemref(rewriter, loc, memref, idx, acc);
        }
        llvm_unreachable("unhandled chain mode");
    }

    // One memory step: load a fresh value from the memref, or store the
    // accumulator back to it (the chain keeps flowing either way).
    Value makeMemref(RewriterBase &rewriter, Location loc, Value memref,
                     Value idx, Value acc) {
        if (std::bernoulli_distribution(0.5)(rng))
            return memref::LoadOp::create(rewriter, loc, memref,
                                          ValueRange{idx});
        memref::StoreOp::create(rewriter, loc, acc, memref, ValueRange{idx});
        return acc;
    }

    Value makeConst(RewriterBase &rewriter, Location loc) {
        return arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(intDist(rng)));
    }

    Value makeArith(RewriterBase &rewriter, Location loc, Value lhs) {
        Value rhs = makeConst(rewriter, loc);
        switch (opDist(rng)) {
        case 0: return arith::AddIOp::create(rewriter, loc, lhs, rhs);
        case 1: return arith::SubIOp::create(rewriter, loc, lhs, rhs);
        default: return arith::MulIOp::create(rewriter, loc, lhs, rhs);
        }
    }

    std::mt19937 &rng;
    std::uniform_int_distribution<int64_t> intDist{-100, 100};
    std::uniform_int_distribution<int> opDist{0, 2};
};

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
            {"load-reordering", &MetamorphicPass::tryLoadReordering},
            {"insert-fence", &MetamorphicPass::tryInsertFence},
            // {"insert-thread", &MetamorphicPass::tryInsertThread},
            {"insert-atomic-write", &MetamorphicPass::tryInsertAtomicWriteInThread},
            {"local-store-duplication", &MetamorphicPass::tryLocalStoreDuplication},
            {"try-insert-random-arith", &MetamorphicPass::tryInsertRandomArith},
            {"try-insert-random-memref", &MetamorphicPass::tryInsertRandomMemref},
            {"try-insert-comparison", &MetamorphicPass::tryInsertComparison},
            {"try-insert-both-arms-if", &MetamorphicPass::tryInsertBothArmsIf},
            {"insert-parallel", &MetamorphicPass::tryInsertParallelOp},
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

        // maximal runs of consecutive loads in each block
        SmallVector<SmallVector<Operation *>> loadRuns =
            collectOpRuns(op, [](Operation *innerOp) {
                return isa<memref::LoadOp>(innerOp);
            });
        for (SmallVector<Operation *> &run : loadRuns) {
            // the whole block shares one thread context
            bool inThread =
                funcInThread || isOpInThread(&run.front()->getBlock()->front());
            SmallVector<memref::LoadOp> loads;
            for (Operation *load : run)
                loads.push_back(cast<memref::LoadOp>(load));
            if (checkLoadRun(loads, inThread))
                eligibleRuns.push_back(std::move(loads));
        }

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
        SmallVector<InsertPoint> points;

        collectInsertPoints(op.getRegion(), points);

        if (points.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
        InsertPoint point = points[dist(rng)];

        if (point.op)
            rewriter.setInsertionPoint(point.op);
        else
            rewriter.setInsertionPointToEnd(point.block);
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

    bool tryInsertAtomicWriteInThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
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

        // Dummy 0-d memref on this thread's stack and an atomic write of 0 to
        // it: a no-op that still exercises the atomic memory path.
        auto memrefType = MemRefType::get({}, rewriter.getI32Type());
        auto dummyMemref = memref::AllocaOp::create(rewriter, loc, memrefType,
                                                    /*dynamicSizes=*/ValueRange{},
                                                    /*alignment=*/IntegerAttr{});
        auto dummyValue = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        omp::AtomicWriteOp::create(rewriter, loc, dummyMemref, dummyValue,
                                   /*hint=*/rewriter.getI64IntegerAttr(0),
                                   /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});

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
            else if (auto atomic = dyn_cast<omp::AtomicWriteOp>(innerOp))
                memref = atomic.getX();
            else
                return;

            if (auto section = innerOp->getParentOfType<omp::SectionOp>())
                memrefToSections[memref].insert(section);
        });

        SmallVector<Operation *> candidates;

        // find all stores (non-atomic stores and atomic writes) that are
        // in a section and whose memref is only accessed in that section.
        // Only thread-local non-atomic stores or atomic writes qualify:
        // duplicating a shared non-atomic store would add a data race.
        op.getBody().walk([&](Operation *innerOp) {
            Value memref;
            bool isAtomicStore = false;
            if (auto storeOp = dyn_cast<memref::StoreOp>(innerOp)) {
                memref = storeOp.getMemref();
            } else if (auto atomic = dyn_cast<omp::AtomicWriteOp>(innerOp)) {
                isAtomicStore = true;
                memref = atomic.getX();
            } else {
                return;
            }

            auto section = innerOp->getParentOfType<omp::SectionOp>();
            if (!section)
                return;

            if (!isAtomicStore && !isThreadLocalMemref(memref, innerOp))
                return;

            // the memref must be only accessed in this section, so the
            // duplicate stays a per-thread mutation of the same location
            auto it = memrefToSections.find(memref);
            if (it == memrefToSections.end() || it->second.size() != 1 ||
                !it->second.contains(section))
                return;

            candidates.push_back(innerOp);
        });

        if (candidates.empty())
            return false;

        // pick one store op at random and duplicate it after itself
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        Operation *store = candidates[dist(rng)];

        rewriter.setInsertionPointAfter(store);
        rewriter.clone(*store);

        return true;
    }

    // Randomly insert a fresh memref and a chain of arith ops
    // Keep going until max chain length of 5 is hit, or a store is inserted back to the memref, which stops the chain
    bool tryInsertRandomArith(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<InsertPoint> points;

        collectInsertPoints(op.getRegion(), points);

        if (points.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
        InsertPoint point = points[dist(rng)];

        if (point.op)
            rewriter.setInsertionPoint(point.op);
        else
            rewriter.setInsertionPointToEnd(point.block);

        ArithGenerator(rng).generate(rewriter, op.getLoc());

        return true;
    }

    // Randomly insert a fresh memref and a chain of loads/stores to it
    bool tryInsertRandomMemref(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<InsertPoint> points;

        collectInsertPoints(op.getRegion(), points);

        if (points.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
        InsertPoint point = points[dist(rng)];

        if (point.op)
            rewriter.setInsertionPoint(point.op);
        else
            rewriter.setInsertionPointToEnd(point.block);

        ArithGenerator(rng).generateMemref(rewriter, op.getLoc());

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

    // Wrap an existing chain of thread-local ops — or, if none exists, a
    // freshly generated memref/arith chain — in an scf.if whose condition
    // cannot be folded, duplicating the ops into both arms. Exactly one arm
    // runs per thread, so semantics are preserved while the lowering of both
    // branches is exercised.
    bool tryInsertBothArmsIf(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        Location loc = op.getLoc();

        // chains of consecutive wrappable ops inside an omp.section; results
        // must be unused so the cloned arm never references values moved into
        // the other arm
        SmallVector<SmallVector<Operation *>> chains =
            collectOpRuns(op, [](Operation *innerOp) {
                return innerOp->getParentOfType<omp::SectionOp>() &&
                       isWrappable(innerOp) &&
                       llvm::all_of(innerOp->getResults(),
                                    [](Value r) { return r.use_empty(); }) &&
                       !isa<omp::BarrierOp, omp::FlushOp, omp::SectionsOp,
                            omp::ParallelOp>(innerOp);
            });

        if (!chains.empty()) {
            std::uniform_int_distribution<size_t> dist(0, chains.size() - 1);
            SmallVector<Operation *> &chain = chains[dist(rng)];

            rewriter.setInsertionPoint(chain.front());
            Value cond = makeRuntimeCondition(chain.front(), rewriter, loc, rng);
            if (!cond)
                cond = arith::ConstantOp::create(
                    rewriter, loc, rewriter.getI1Type(),
                    rewriter.getBoolAttr(true));
            scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, cond,
                                               /*withElseRegion=*/true);
            Block *thenBlock = &ifOp.getThenRegion().front();
            Block *elseBlock = &ifOp.getElseRegion().front();

            // the if builder already terminated both regions with scf.yield,
            // so only append one when a block came out unterminated
            auto ensureYield = [&](Block *block) {
                if (block->empty() ||
                    !block->back().hasTrait<OpTrait::IsTerminator>()) {
                    rewriter.setInsertionPointToEnd(block);
                    scf::YieldOp::create(rewriter, loc, ValueRange{});
                }
            };
            Operation *thenTerm = thenBlock->getTerminator();
            for (Operation *chainOp : chain)
                if (thenTerm)
                    chainOp->moveBefore(thenTerm);
                else
                    chainOp->moveBefore(thenBlock, thenBlock->end());
            Operation *elseTerm = elseBlock->getTerminator();
            if (elseTerm)
                rewriter.setInsertionPoint(elseTerm);
            else
                rewriter.setInsertionPointToEnd(elseBlock);
            for (Operation *chainOp : chain)
                rewriter.clone(*chainOp);
            ensureYield(thenBlock);
            ensureYield(elseBlock);
            return true;
        }

        // fallback: a random memref/arith chain in both arms of a random
        // non-empty section
        SmallVector<omp::SectionOp> sections;
        op.getBody().walk([&](omp::SectionOp sectionOp) {
            if (!sectionOp.getRegion().empty() &&
                !sectionOp.getRegion().front().empty())
                sections.push_back(sectionOp);
        });
        if (sections.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, sections.size() - 1);
        Block &sectionBlock = sections[dist(rng)].getRegion().front();
        Operation *anchor = sectionBlock.getTerminator();
        if (!anchor)
            return false;

        rewriter.setInsertionPoint(anchor);
        Value cond = makeRuntimeCondition(anchor, rewriter, loc, rng);
        if (!cond)
            cond = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI1Type(),
                rewriter.getBoolAttr(true));
        scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, cond,
                                           /*withElseRegion=*/true);
        Block *thenBlock = &ifOp.getThenRegion().front();
        Block *elseBlock = &ifOp.getElseRegion().front();

        ArithGenerator gen(rng);
        auto genChain = [&]() {
            switch (std::uniform_int_distribution<int>(0, 2)(rng)) {
            case 0: gen.generate(rewriter, loc); break;
            case 1: gen.generateMemref(rewriter, loc); break;
            default: gen.generateMixed(rewriter, loc); break;
            }
        };
        rewriter.setInsertionPointToStart(thenBlock);
        genChain();
        rewriter.setInsertionPointToStart(elseBlock);
        genChain();
        auto ensureYield = [&](Block *block) {
            if (block->empty() ||
                !block->back().hasTrait<OpTrait::IsTerminator>()) {
                rewriter.setInsertionPointToEnd(block);
                scf::YieldOp::create(rewriter, loc, ValueRange{});
            }
        };
        ensureYield(thenBlock);
        ensureYield(elseBlock);
        return true;
    }

    bool tryInsertCriticalOp(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        // maximal runs of consecutive wrappable ops inside an omp.section, so
        // each chain is a thread-local group of ops that can be moved into an
        // omp.critical as one unit
        SmallVector<SmallVector<Operation *>> chains =
            collectOpRuns(op, [](Operation *innerOp) {
                return innerOp->getParentOfType<omp::SectionOp>() &&
                       isWrappable(innerOp);
            });

        if (chains.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, chains.size() - 1);
        SmallVector<Operation *> &chain = chains[dist(rng)];

        rewriter.setInsertionPoint(chain.front());
        omp::CriticalOp critical = omp::CriticalOp::create(
            rewriter, op.getLoc(), /*name=*/FlatSymbolRefAttr{});

        Block *body = rewriter.createBlock(&critical.getRegion());
        for (Operation *chainOp : chain)
            chainOp->moveBefore(body, body->end());
        rewriter.setInsertionPointToEnd(body);
        omp::TerminatorOp::create(rewriter, op.getLoc());

        return true;
    }

    // Insert a new omp.parallel right after a randomly chosen existing one.
    // The new parallel contains an omp.sections with a random number of
    // omp.section bodies, each filled with the combined arith/memref chain.
    bool tryInsertParallelOp(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<omp::ParallelOp> candidates;
        op.getBody().walk([&](omp::ParallelOp parallel) {
            candidates.push_back(parallel);
        });

        if (candidates.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        omp::ParallelOp anchor = candidates[dist(rng)];

        rewriter.setInsertionPointAfter(anchor);
        Location loc = anchor.getLoc();

        omp::ParallelOp parallel = omp::ParallelOp::create(rewriter, loc);
        Block *parallelBody = rewriter.createBlock(&parallel.getRegion());

        omp::SectionsOp sections =
            omp::SectionsOp::create(rewriter, loc, TypeRange{}, ValueRange{});
        Block *sectionsBlock = rewriter.createBlock(&sections.getRegion());

        ArithGenerator arithGen(rng);
        int numSections = std::uniform_int_distribution<int>(1, 4)(rng);
        for (int i = 0; i < numSections; ++i) {
            omp::SectionOp section = omp::SectionOp::create(rewriter, loc);
            Block *sectionBody = rewriter.createBlock(&section.getRegion());
            arithGen.generateMixed(rewriter, loc);
            rewriter.setInsertionPointToEnd(sectionBody);
            omp::TerminatorOp::create(rewriter, loc);
            rewriter.setInsertionPointToEnd(sectionsBlock);
        }
        omp::TerminatorOp::create(rewriter, loc);
        rewriter.setInsertionPointToEnd(parallelBody);
        omp::TerminatorOp::create(rewriter, loc);

        return true;
    }

    bool tryUnrollSingleParallelOp(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

        // Find all omp.parallel ops in the function body that contain exactly
        // one omp.sections op, which in turn contains exactly one omp.section
        // op. Only direct children count, so nested sections ops inside a
        // section body do not disqualify the parallel.
        SmallVector<omp::ParallelOp> candidates;
        op.getBody().walk([&](omp::ParallelOp parallel) {
            Region &parallelRegion = parallel.getRegion();
            if (parallelRegion.getBlocks().size() != 1)
                return;
            Block &parallelBlock = parallelRegion.front();
            auto sectionsOps =
                llvm::to_vector(parallelBlock.getOps<omp::SectionsOp>());
            if (sectionsOps.size() != 1)
                return;
            omp::SectionsOp sections = sectionsOps[0];
            Region &sectionsRegion = sections.getRegion();
            if (sectionsRegion.getBlocks().size() != 1)
                return;
            Block &sectionsBlock = sectionsRegion.front();
            auto sectionOps =
                llvm::to_vector(sectionsBlock.getOps<omp::SectionOp>());
            if (sectionOps.size() != 1)
                return;
            candidates.push_back(parallel);
        });
        
        if (candidates.empty())
            return false;

        // Unroll every eligible parallel: move the single section's body to
        // where the omp.parallel op was, then remove the section, sections,
        // and parallel ops.
        bool unrolled = false;
        for (omp::ParallelOp parallel : candidates) {
            // re-locate the single omp.sections/omp.section inside the parallel
            omp::SectionsOp sections;
            parallel.walk([&](omp::SectionsOp s) { sections = s; });
            omp::SectionOp section;
            sections.walk([&](omp::SectionOp s) { section = s; });

            // Only unroll a bare nest: no clause operands or block arguments
            // (private/reduction vars), no stray ops beside the sections/section,
            // and a single-block section body so it moves out as one unit.
            if (parallel->getNumOperands() != 0 || parallel->getNumResults() != 0)
                continue;
            if (sections->getParentOp() != parallel.getOperation() ||
                section->getParentOp() != sections.getOperation())
                continue;
            Block &parallelBlock = parallel.getRegion().front();
            Block &sectionsBlock = sections.getRegion().front();
            if (parallelBlock.getArguments().size() != 0 ||
                sectionsBlock.getArguments().size() != 0 ||
                parallelBlock.getOperations().size() != 2 ||
                sectionsBlock.getOperations().size() != 2)
                continue;
            if (section.getRegion().empty() ||
                section.getRegion().getBlocks().size() != 1)
                continue;

            // Move the section body in front of the omp.parallel, skipping its
            // terminator, then remove the section, sections, and parallel ops.
            Block &sectionBlock = section.getRegion().front();
            Operation *terminator = sectionBlock.getTerminator();
            rewriter.setInsertionPoint(parallel);
            for (Operation &op : llvm::make_early_inc_range(sectionBlock))
                if (&op != terminator)
                    rewriter.insert(&op);

            rewriter.eraseOp(section);
            rewriter.eraseOp(sections);
            rewriter.eraseOp(parallel);

            unrolled = true;
        }

        return unrolled;
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
#include <stack>
