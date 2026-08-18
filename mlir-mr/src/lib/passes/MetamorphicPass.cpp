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
#include "mlir/IR/Dominance.h"
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
#include <optional>
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

// Position the rewriter at the given insertion point.
static void setInsertPoint(RewriterBase &rewriter, const InsertPoint &point) {
    if (point.op)
        rewriter.setInsertionPoint(point.op);
    else
        rewriter.setInsertionPointToEnd(point.block);
}

// Collect legal insertion positions inside thread regions (omp.section /
// omp.parallel bodies). Blocks that are direct children of omp.sections
// regions are excluded, since those regions only allow omp.section ops and a
// terminator. Empty blocks and blocks that do not end in a terminator are
// included, so generated ops can also land in empty bodies.
static void collectThreadInsertPoints(Region &region, bool inThread,
                                      SmallVectorImpl<InsertPoint> &points) {
    for (Block &block : region) {
        bool blockInThread = inThread ||
                             isa<omp::SectionOp>(block.getParentOp()) ||
                             isa<omp::ParallelOp>(block.getParentOp());
        if (!isa<omp::SectionsOp>(block.getParentOp())) {
            if (blockInThread) {
                for (Operation &op : block)
                    points.push_back({&op, nullptr});
                if (block.empty() ||
                    !block.back().hasTrait<OpTrait::IsTerminator>())
                    points.push_back({nullptr, &block});
            }
        }
        for (Operation &op : block)
            for (Region &child : op.getRegions())
                collectThreadInsertPoints(child, blockInThread, points);
    }
}

// Returns a random insertion point inside a thread region. When the function
// has no thread region, a fresh omp.parallel / omp.sections / omp.section nest
// is created at the start of the entry block, so code-inserting transforms
// always land inside a thread.
static InsertPoint makeThreadInsertPoint(func::FuncOp op,
                                         RewriterBase &rewriter,
                                         std::mt19937 &rng) {
    SmallVector<InsertPoint> points;
    collectThreadInsertPoints(op.getRegion(), /*inThread=*/false, points);
    if (!points.empty()) {
        std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
        return points[dist(rng)];
    }

    if (op.getBody().empty())
        return {nullptr, nullptr};

    Location loc = op.getLoc();
    rewriter.setInsertionPointToStart(&op.getBody().front());

    omp::ParallelOp parallel = omp::ParallelOp::create(rewriter, loc);
    Block *parallelBody = rewriter.createBlock(&parallel.getRegion());
    omp::SectionsOp sections =
        omp::SectionsOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    Block *sectionsBlock = rewriter.createBlock(&sections.getRegion());
    omp::SectionOp section = omp::SectionOp::create(rewriter, loc);
    Block *sectionBody = rewriter.createBlock(&section.getRegion());
    rewriter.setInsertionPointToEnd(sectionBody);
    omp::TerminatorOp::create(rewriter, loc);
    Operation *anchor = sectionBody->getTerminator();

    rewriter.setInsertionPointToEnd(sectionsBlock);
    omp::TerminatorOp::create(rewriter, loc);
    rewriter.setInsertionPointToEnd(parallelBody);
    omp::TerminatorOp::create(rewriter, loc);

    return {anchor, nullptr};
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

// True if a section body behaves identically when executed once by a single
// thread: it touches no shared memory (every memref operand is thread-local)
// and contains no team-level OpenMP constructs or calls. A single-section
// omp.sections runs that section on exactly one team thread, so unrolling it
// out of the omp.parallel preserves the post-join state only under these
// conditions.
static bool isSingleThreadBody(Region &region) {
    WalkResult result = region.walk([&](Operation *innerOp) {
        if (isa<omp::BarrierOp, omp::CriticalOp, omp::FlushOp, omp::MasterOp,
                omp::MaskedOp, omp::OrderedOp, omp::ScopeOp, omp::SingleOp,
                omp::TaskOp, omp::TaskgroupOp, omp::TaskwaitOp,
                omp::TaskyieldOp, omp::ThreadprivateOp, omp::WsloopOp,
                omp::ParallelOp, omp::SectionsOp, omp::SectionOp>(innerOp))
            return WalkResult::interrupt();
        if (isa<func::CallOp>(innerOp))
            return WalkResult::interrupt();
        for (Value operand : innerOp->getOperands())
            if (isa<MemRefType>(operand.getType()) &&
                !isThreadLocalMemref(operand, innerOp))
                return WalkResult::interrupt();
        return WalkResult::advance();
    });
    return !result.wasInterrupted();
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

// compose two applied relations in application order
static mlir_mr::OutcomeRelation composeRelation(mlir_mr::OutcomeRelation a,
                                                mlir_mr::OutcomeRelation b) {
    if (a == mlir_mr::OutcomeRelation::Equality)
        return b;
    if (b == mlir_mr::OutcomeRelation::Equality)
        return a;
    if (a == b)
        return a;
    llvm_unreachable("mixing subset and superset transforms in one run");
}

// true if a transform of relation `next` may follow one of relation `cur`:
// once a direction is fixed, only transforms preserving or extending it may
// apply
static bool canApplyAfter(mlir_mr::OutcomeRelation cur,
                          mlir_mr::OutcomeRelation next) {
    if (cur == mlir_mr::OutcomeRelation::Equality)
        return true;
    return cur == next;
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

        // each transform declares the outcome-set relation it preserves; the
        // comparison direction  for a run follows the composition of the
        // transforms actually applied (see core.cpp)
        struct TransformSpec {
            const char *name;
            Transform fn;
            mlir_mr::OutcomeRelation relation;
        };

        static const TransformSpec kTransforms[] = {
            {"load-reordering", &MetamorphicPass::tryLoadReordering,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-fence", &MetamorphicPass::tryInsertFence,
             mlir_mr::OutcomeRelation::Subset},
            {"remove-fence", &MetamorphicPass::tryRemoveFence,
             mlir_mr::OutcomeRelation::Superset},
            // {"insert-thread", &MetamorphicPass::tryInsertThread},
            {"insert-atomic-write", &MetamorphicPass::tryInsertAtomicWriteInThread,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-atomic-read", &MetamorphicPass::tryInsertAtomicReadInThread,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-atomic-cas", &MetamorphicPass::tryInsertAtomicCAS,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-read-arith", &MetamorphicPass::tryInsertReadArith,
             mlir_mr::OutcomeRelation::Equality},
            {"local-store-duplication", &MetamorphicPass::tryLocalStoreDuplication,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-random-arith", &MetamorphicPass::tryInsertRandomArith,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-random-memref", &MetamorphicPass::tryInsertRandomMemref,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-comparison", &MetamorphicPass::tryInsertComparison,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-both-arms-if", &MetamorphicPass::tryInsertBothArmsIf,
             mlir_mr::OutcomeRelation::Equality},
            {"insert-parallel", &MetamorphicPass::tryInsertParallelOp,
             mlir_mr::OutcomeRelation::Equality},
            {"relax-operation", &MetamorphicPass::tryRelaxOperation,
             mlir_mr::OutcomeRelation::Superset},
            {"restrict-operation", &MetamorphicPass::tryRestrictOperation,
             mlir_mr::OutcomeRelation::Subset},
            {"insert-fence-between-mem-ops", &MetamorphicPass::tryInsertFenceInbetweenMemOps,
            mlir_mr::OutcomeRelation::Subset},
            {"insert-critical", &MetamorphicPass::tryInsertCriticalSection,
            mlir_mr::OutcomeRelation::Subset},
            {"unroll-single-thread", &MetamorphicPass::tryUnrollSingleThread,
            mlir_mr::OutcomeRelation::Equality},
        };

        auto findSpec = [](llvm::StringRef name) -> const TransformSpec * {
            for (const auto &spec : kTransforms)
                if (spec.name == name)
                    return &spec;
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
        SmallVector<const TransformSpec *> transforms;
        for (const auto &spec : kTransforms)
            if (requestedNames.empty() ||
                llvm::is_contained(requestedNames, spec.name))
                transforms.push_back(&spec);
        std::sort(transforms.begin(), transforms.end(),
                  [](const TransformSpec *a, const TransformSpec *b) {
                      return std::string(a->name) < std::string(b->name);
                  });
        if (runInfo)
            for (llvm::StringRef name : requestedNames)
                runInfo->requestedTransforms.push_back(name.str());

        // collect all functions in the module
        SmallVector<func::FuncOp> funcs;
        op.walk([&](func::FuncOp f) { funcs.push_back(f); });
        if (funcs.empty())
            return;

        // pick a function at random to transform; if no transformation
        // applies, repick a new test case (function) up to a fixed number of
        // attempts before giving up
        std::uniform_int_distribution<size_t> dist(0, funcs.size() - 1);
        constexpr int kMaxTransformAttempts = 3;
        llvm::DenseSet<func::FuncOp> tried;
        for (int attempt = 0; attempt < kMaxTransformAttempts; ++attempt) {
            func::FuncOp target;
            do {
                target = funcs[dist(rng)];
            } while (tried.contains(target) && tried.size() < funcs.size());
            tried.insert(target);

            int transformCounter = 0;
            std::optional<mlir_mr::OutcomeRelation> aggregate;

            // keep applying random transformations until max applications is
            // reached; transforms whose relation would contradict the direction
            // already established are excluded from the draw
            while (transformCounter < maxApply) {
                SmallVector<const TransformSpec *> allowed;
                for (const TransformSpec *spec : transforms)
                    if (canApplyAfter(
                            aggregate.value_or(mlir_mr::OutcomeRelation::Equality),
                            spec->relation))
                        allowed.push_back(spec);
                if (allowed.empty())
                    break;

                std::shuffle(allowed.begin(), allowed.end(), rng);
                bool applied = false;
                for (const TransformSpec *spec : allowed)
                    if ((this->*spec->fn)(target, rewriter, rng)) {
                        transformCounter++;
                        aggregate =
                            aggregate
                                ? composeRelation(*aggregate, spec->relation)
                                : spec->relation;
                        if (runInfo) {
                            runInfo->appliedTransforms.push_back(
                                {spec->name, target.getName().str()});
                            runInfo->transformApplied = true;
                        }
                        llvm::errs() << "=== AFTER " << spec->name << " ===\n";
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

        if (runInfo)
            runInfo->error = "tried applying transformation " +
                             std::to_string(kMaxTransformAttempts) + " times";
        signalPassFailure();
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
            // only reorder loads that actually run inside a thread region
            bool inThread =
                funcInThread || isOpInThread(&run.front()->getBlock()->front());
            if (!inThread)
                continue;
            SmallVector<memref::LoadOp> loads;
            for (Operation *load : run)
                loads.push_back(cast<memref::LoadOp>(load));
            if (checkLoadRun(loads, /*requireLocalMemrefs=*/true))
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

    // randomly insert an omp.flush fence inside a thread region, creating a
    // thread region when none exists
    bool tryInsertFence(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);
        omp::FlushOp::create(rewriter, op.getLoc(), ValueRange{});
        return true;
    }

    // randomly remove fence
    bool tryRemoveFence(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        SmallVector<omp::FlushOp> fences;
        op.walk([&](omp::FlushOp fence) { fences.push_back(fence); });
        if (fences.empty())
            return false;
        std::uniform_int_distribution<size_t> dist(0, fences.size() - 1);
        rewriter.eraseOp(fences[dist(rng)]);
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
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);

        Location loc = op.getLoc();

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

    bool tryInsertAtomicReadInThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);

        Location loc = op.getLoc();

        mlir::DominanceInfo domInfo(op);
        auto dominatesPoint = [&](Operation *defOp) {
            if (point.op)
                return domInfo.properlyDominates(defOp, point.op);
            // appending at the end of a block: a def in that block or in an
            // enclosing region is in scope
            return defOp->getBlock() == point.block ||
                   defOp->getParentRegion()->isAncestor(
                       point.block->getParent());
        };
        SmallVector<Value> candidates;
        op.getBody().walk([&](Operation *innerOp) {
            if (!isOpInThread(innerOp) ||
                innerOp->hasTrait<OpTrait::IsTerminator>())
                return;
            for (Value res : innerOp->getResults())
                if (isa<MemRefType>(res.getType()) &&
                    dominatesPoint(res.getDefiningOp()))
                    candidates.push_back(res);
        });
        if (candidates.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        Value targetMemref = candidates[dist(rng)];
        auto memrefType = cast<MemRefType>(targetMemref.getType());
        Value destMemref = memref::AllocaOp::create(
            rewriter, loc, memrefType, /*dynamicSizes=*/ValueRange{},
            /*alignment=*/IntegerAttr{});

        omp::AtomicReadOp::create(rewriter, loc, targetMemref, destMemref,
                                  memrefType.getElementType(),
                                  /*hint=*/0,
                                  omp::ClauseMemoryOrderKindAttr{});

        return true;
    }

    // Insert a strong compare-and-swap on a fresh thread-local location. The
    // location is atomically initialised to 0 and the CAS expects 42, so it
    // never fires: the value stays 0 and the post-join state is unchanged,
    // while the cmpxchg path of omp.atomic.compare is exercised.
    bool tryInsertAtomicCAS(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);

        Location loc = op.getLoc();

        auto memrefType = MemRefType::get({}, rewriter.getI32Type());
        Value memref = memref::AllocaOp::create(rewriter, loc, memrefType,
                                                /*dynamicSizes=*/ValueRange{},
                                                /*alignment=*/IntegerAttr{});
        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        omp::AtomicWriteOp::create(rewriter, loc, memref, zero,
                                   /*hint=*/rewriter.getI64IntegerAttr(0),
                                   /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});

        omp::AtomicCompareOp cas = omp::AtomicCompareOp::create(
            rewriter, loc, memref, /*weak=*/false, /*hint=*/0,
            /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});
        Block *body = rewriter.createBlock(&cas.getRegion(), {},
                                           {rewriter.getI32Type()}, {loc});
        rewriter.setInsertionPointToStart(body);
        Value expected = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(42));
        Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                          body->getArgument(0), expected);
        Value desired = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(7));
        Value sel = arith::SelectOp::create(rewriter, loc, cmp, desired,
                                            body->getArgument(0));
        omp::YieldOp::create(rewriter, loc, ValueRange{sel});
        return true;
    }

    bool tryInsertReadArith(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        // get all atomic writes in the function and pick one at random
        SmallVector<omp::AtomicWriteOp> writes;
        op.getBody().walk([&](omp::AtomicWriteOp write) {
            writes.push_back(write);
        });

        if (writes.empty())
            return false;

        // only writes of an integer or index expression can carry the no-op
        // arith wrapper; pointer-typed expressions (e.g. a memref) cannot
        SmallVector<omp::AtomicWriteOp> intWrites;
        for (auto write : writes) {
            if (write.getExpr().getType().isIntOrIndex())
                intWrites.push_back(write);
        }

        if (intWrites.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, intWrites.size() - 1);
        omp::AtomicWriteOp targetWrite = intWrites[dist(rng)];

        // insert a random add or sub 0 on the stored expression before the
        // atomic write, keeping the stored value unchanged
        rewriter.setInsertionPoint(targetWrite);
        Location loc = targetWrite.getLoc();
        Value value = targetWrite.getExpr();
        Value zero = arith::ConstantOp::create(rewriter, loc, value.getType(),
            IntegerAttr::get(value.getType(), 0));
        Value newValue;
        if (std::bernoulli_distribution(0.5)(rng)) {
            newValue = arith::AddIOp::create(rewriter, loc, value, zero);
        } else {
            newValue = arith::SubIOp::create(rewriter, loc, value, zero);
        }
        targetWrite.setOperand(1, newValue);

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

    // Randomly insert a fresh memref and a chain of arith ops inside a thread
    // region, creating one when none exists
    bool tryInsertRandomArith(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);

        ArithGenerator(rng).generate(rewriter, op.getLoc());

        return true;
    }

    // Randomly insert a fresh memref and a chain of loads/stores to it inside
    // a thread region, creating one when none exists
    bool tryInsertRandomMemref(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        setInsertPoint(rewriter, point);

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
                omp::SectionOp sectionOp =
                    innerOp->getParentOfType<omp::SectionOp>();
                return sectionOp &&
                       innerOp->getParentOp() == sectionOp.getOperation() &&
                       isWrappable(innerOp) &&
                       llvm::all_of(innerOp->getResults(),
                                    [](Value r) { return r.use_empty(); }) &&
                       !isa<omp::BarrierOp, omp::FlushOp, omp::SectionsOp,
                            omp::ParallelOp, omp::AtomicCompareOp>(innerOp);
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

        // fallback: a random memref/arith chain in both arms of an scf.if
        // inside a thread region, creating one when none exists
        InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
        if (!point.op && !point.block)
            return false;
        Operation *anchor = point.op ? point.op : point.block->getTerminator();
        setInsertPoint(rewriter, point);
        Value cond = anchor ? makeRuntimeCondition(anchor, rewriter, loc, rng)
                            : Value{};
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

    bool tryInsertCriticalSection(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
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

    // Insert a new omp.parallel after a randomly chosen existing one, or at
    // the start of the entry block when none exists. The new parallel contains
    // an omp.sections with a random number of omp.section bodies, each filled
    // with the combined arith/memref chain.
    bool tryInsertParallelOp(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        if (op.getBody().empty())
            return false;

        SmallVector<omp::ParallelOp> candidates;
        op.getBody().walk([&](omp::ParallelOp parallel) {
            candidates.push_back(parallel);
        });

        if (!candidates.empty()) {
            std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
            rewriter.setInsertionPointAfter(candidates[dist(rng)]);
        } else {
            rewriter.setInsertionPointToStart(&op.getBody().front());
        }
        Location loc = op.getLoc();

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

    bool tryUnrollSingleThread(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {

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
            if (sections->getNumOperands() != 0 || sections->getNumResults() != 0)
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

            // The unrolled body runs once on the main thread, so it must be
            // practically equivalent to a single-thread execution: thread-local
            // memory only, no team-level constructs, and at least one real op.
            Block &sectionBlock = section.getRegion().front();
            if (sectionBlock.getOperations().size() < 2 ||
                !isSingleThreadBody(section.getRegion()))
                continue;

            // Move the section body in front of the omp.parallel, skipping its
            // terminator, then remove the section, sections, and parallel ops.
            Operation *terminator = sectionBlock.getTerminator();
            for (Operation &op : llvm::make_early_inc_range(sectionBlock))
                if (&op != terminator)
                    rewriter.moveOpBefore(&op, parallel.getOperation());

            rewriter.eraseOp(section);
            rewriter.eraseOp(sections);
            rewriter.eraseOp(parallel);

            unrolled = true;
        }

        return unrolled;
    }

    // Relax the memory order of a randomly chosen atomic op by exactly one
    // stage. Only ops that can actually be weakened are candidates: an op
    // already at the weakest order (relaxed) has no successor and is skipped.
    bool tryRelaxOperation(func::FuncOp op, RewriterBase &rewriter,
                           std::mt19937 &rng) {
        // Valid one-stage weakening chains per op kind, strongest first.
        // Orders outside the chain (e.g. acquire on a write) cannot be
        // weakened meaningfully and are left alone.
        auto nextWeakerOrder = [&](Operation *innerOp,
                                  omp::ClauseMemoryOrderKind order)
            -> std::optional<omp::ClauseMemoryOrderKind> {
            if (isa<omp::AtomicWriteOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Seq_cst:
                    return omp::ClauseMemoryOrderKind::Release;
                case omp::ClauseMemoryOrderKind::Release:
                    return omp::ClauseMemoryOrderKind::Relaxed;
                default:
                    return std::nullopt;
                }
            }
            if (isa<omp::AtomicReadOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Seq_cst:
                    return omp::ClauseMemoryOrderKind::Acquire;
                case omp::ClauseMemoryOrderKind::Acquire:
                    return omp::ClauseMemoryOrderKind::Relaxed;
                default:
                    return std::nullopt;
                }
            }
            if (isa<omp::AtomicCompareOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Seq_cst:
                    return std::uniform_int_distribution<int>(0, 2)(rng) == 0
                               ? omp::ClauseMemoryOrderKind::Acq_rel
                               : (std::uniform_int_distribution<int>(0, 1)(rng) == 0
                                      ? omp::ClauseMemoryOrderKind::Acquire
                                      : omp::ClauseMemoryOrderKind::Release);
                case omp::ClauseMemoryOrderKind::Acq_rel:
                    return std::uniform_int_distribution<int>(0, 1)(rng) == 0
                               ? omp::ClauseMemoryOrderKind::Acquire
                               : omp::ClauseMemoryOrderKind::Release;
                case omp::ClauseMemoryOrderKind::Acquire:
                    return omp::ClauseMemoryOrderKind::Relaxed;
                default:
                    return std::nullopt;
                }
            }
            return std::nullopt;
        };

        SmallVector<std::pair<Operation *, omp::ClauseMemoryOrderKind>>
            candidates;
        op.getBody().walk([&](Operation *innerOp) {
            std::optional<omp::ClauseMemoryOrderKind> order;
            if (auto write = dyn_cast<omp::AtomicWriteOp>(innerOp))
                order = write.getMemoryOrder();
            else if (auto read = dyn_cast<omp::AtomicReadOp>(innerOp))
                order = read.getMemoryOrder();
            else
                return;

            // memory_order defaults to seq_cst when not specified
            omp::ClauseMemoryOrderKind current =
                order.value_or(omp::ClauseMemoryOrderKind::Seq_cst);
            if (auto weaker = nextWeakerOrder(innerOp, current))
                candidates.push_back({innerOp, *weaker});
        });

        if (candidates.empty())
            return false;
        
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        auto [opToRelax, weakerOrder] = candidates[dist(rng)];

        if (auto write = dyn_cast<omp::AtomicWriteOp>(opToRelax))
            write.setMemoryOrder(weakerOrder);
        else if (auto read = dyn_cast<omp::AtomicReadOp>(opToRelax))
            read.setMemoryOrder(weakerOrder);

        return true;
    }

    bool tryRestrictOperation(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        // Valid one-stage weakening chains per op kind, strongest first.
        // Orders outside the chain (e.g. acquire on a write) cannot be
        // weakened meaningfully and are left alone.
        auto nextStrongerOrder = [&](Operation *innerOp,
                                  omp::ClauseMemoryOrderKind order)
            -> std::optional<omp::ClauseMemoryOrderKind> {
            if (isa<omp::AtomicWriteOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Relaxed:
                    return omp::ClauseMemoryOrderKind::Release;
                case omp::ClauseMemoryOrderKind::Release:
                    return omp::ClauseMemoryOrderKind::Seq_cst;
                default:
                    return std::nullopt;
                }
            }
            if (isa<omp::AtomicReadOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Relaxed:
                    return omp::ClauseMemoryOrderKind::Acquire;
                case omp::ClauseMemoryOrderKind::Acquire:
                    return omp::ClauseMemoryOrderKind::Seq_cst;
                default:
                    return std::nullopt;
                }
            }
            if (isa<omp::AtomicCompareOp>(innerOp)) {
                switch (order) {
                case omp::ClauseMemoryOrderKind::Relaxed:
                    // pick between acquire, acq_rel or seq_cst
                    return std::uniform_int_distribution<int>(0, 2)(rng) == 0
                               ? omp::ClauseMemoryOrderKind::Acquire
                               : (std::uniform_int_distribution<int>(0, 1)(rng) == 0
                                      ? omp::ClauseMemoryOrderKind::Acq_rel
                                      : omp::ClauseMemoryOrderKind::Seq_cst);
                case omp::ClauseMemoryOrderKind::Acquire:
                    return std::uniform_int_distribution<int>(0, 1)(rng) == 0
                               ? omp::ClauseMemoryOrderKind::Acq_rel
                               : omp::ClauseMemoryOrderKind::Seq_cst;
                case omp::ClauseMemoryOrderKind::Acq_rel:
                    return omp::ClauseMemoryOrderKind::Seq_cst;
                default:
                    return std::nullopt;
                }
            }
            return std::nullopt;
        };

        SmallVector<std::pair<Operation *, omp::ClauseMemoryOrderKind>>
            candidates;
        op.getBody().walk([&](Operation *innerOp) {
            std::optional<omp::ClauseMemoryOrderKind> order;
            if (auto write = dyn_cast<omp::AtomicWriteOp>(innerOp))
                order = write.getMemoryOrder();
            else if (auto read = dyn_cast<omp::AtomicReadOp>(innerOp))
                order = read.getMemoryOrder();
            else
                return;

            // memory_order defaults to seq_cst when not specified
            omp::ClauseMemoryOrderKind current =
                order.value_or(omp::ClauseMemoryOrderKind::Seq_cst);
            if (auto stronger = nextStrongerOrder(innerOp, current))
                candidates.push_back({innerOp, *stronger});
        });

        if (candidates.empty())
            return false;
        
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        auto [opToRestrict, strongerOrder] = candidates[dist(rng)];

        if (auto write = dyn_cast<omp::AtomicWriteOp>(opToRestrict))
            write.setMemoryOrder(strongerOrder);
        else if (auto read = dyn_cast<omp::AtomicReadOp>(opToRestrict))
            read.setMemoryOrder(strongerOrder);

        return true;
    }

    // ARMv8 TRANSFORMS BELOW

    bool tryInsertFenceInbetweenMemOps(func::FuncOp op, RewriterBase &rewriter, std::mt19937 &rng) {
        // get chains of consecutive atomic writes and reads in threads
        SmallVector<SmallVector<Operation *>> chains =
            collectOpRuns(op, [](Operation *innerOp) {
                return isOpInThread(innerOp) &&
                       (isa<omp::AtomicWriteOp>(innerOp) ||
                        isa<omp::AtomicReadOp>(innerOp));
            });
        
        // only consider chains of length >= 2, as we need to insert a fence inbetween two ops
        llvm::erase_if(chains, [](const SmallVector<Operation *> &chain) {
            return chain.size() < 2;
        });

        if (chains.empty())
            return false;

        std::uniform_int_distribution<size_t> dist(0, chains.size() - 1);
        SmallVector<Operation *> &chain = chains[dist(rng)];

        rewriter.setInsertionPointAfter(chain[0]);
        omp::FlushOp::create(rewriter, op.getLoc(), ValueRange{});
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
#include <stack>
