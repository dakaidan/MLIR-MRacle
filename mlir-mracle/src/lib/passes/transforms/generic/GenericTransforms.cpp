#include "mlir-mracle/passes/transforms/generic/GenericTransforms.h"
#include "mlir-mracle/passes/transforms/Transforms.h"
#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "mlir-mracle/context/context.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iterator>
#include <optional>
#include <random>
#include <utility>

namespace mlir {
namespace metamorphic {
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

// A shared-memory op inside a thread region: an atomic access or a flush.
// Jitter delays are inserted in the gaps around these ops so the shared
// accesses of different threads drift apart instead of firing in lockstep.
static bool isSharedMemoryOp(Operation *op) {
    return isa<omp::AtomicWriteOp, omp::AtomicReadOp, omp::AtomicCompareOp,
                omp::FlushOp>(op);
}

// Collect the gaps between consecutive shared-memory ops inside thread
// regions: one insert point before every shared-memory op (covering the
// before-first and between gaps) plus one after the last shared-memory op in
// each block that has any.
static void collectSharedMemoryGaps(Region &region, bool inThread,
                                    SmallVectorImpl<InsertPoint> &gaps) {
    for (Block &block : region) {
        bool blockInThread = inThread ||
                             isa<omp::SectionOp>(block.getParentOp()) ||
                             isa<omp::ParallelOp>(block.getParentOp());
        if (!isa<omp::SectionsOp>(block.getParentOp()) && blockInThread) {
            SmallVector<Operation *> shared;
            for (Operation &op : block)
                if (isSharedMemoryOp(&op))
                    shared.push_back(&op);
            for (Operation *sharedOp : shared)
                gaps.push_back({sharedOp, nullptr});
            if (!shared.empty()) {
                if (block.back().hasTrait<OpTrait::IsTerminator>())
                    gaps.push_back({&block.back(), nullptr});
                else
                    gaps.push_back({nullptr, &block});
            }
        }
        for (Operation &op : block)
            for (Region &child : op.getRegions())
                collectSharedMemoryGaps(child, blockInThread, gaps);
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

} // namespace

namespace {

// Direction in which an atomic op's memory order may be stepped.
enum class MemoryOrderDirection { Weaker, Stronger };

// Returns the effective memory order of an atomic read/write/compare op
// (seq_cst when the attribute is unset), or nullopt when the op is not
// atomic.
static std::optional<omp::ClauseMemoryOrderKind>
getMemoryOrder(Operation *innerOp) {
    if (auto write = dyn_cast<omp::AtomicWriteOp>(innerOp))
        return write.getMemoryOrder().value_or(
            omp::ClauseMemoryOrderKind::Seq_cst);
    if (auto read = dyn_cast<omp::AtomicReadOp>(innerOp))
        return read.getMemoryOrder().value_or(
            omp::ClauseMemoryOrderKind::Seq_cst);
    if (auto cmp = dyn_cast<omp::AtomicCompareOp>(innerOp))
        return cmp.getMemoryOrder().value_or(
            omp::ClauseMemoryOrderKind::Seq_cst);
    return std::nullopt;
}

// Writes back the memory order of an atomic read/write op.
static void setMemoryOrder(Operation *op, omp::ClauseMemoryOrderKind order) {
    if (auto write = dyn_cast<omp::AtomicWriteOp>(op))
        write.setMemoryOrder(order);
    else if (auto read = dyn_cast<omp::AtomicReadOp>(op))
        read.setMemoryOrder(order);
}

// The next weaker order for an op, strongest first. Orders outside the
// chain (e.g. acquire on a write) cannot be weakened meaningfully and are
// left alone.
static std::optional<omp::ClauseMemoryOrderKind>
nextWeakerOrder(Operation *innerOp, omp::ClauseMemoryOrderKind order,
                std::mt19937 &rng) {
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
}

// The next stronger order for an op, weakest first. Orders outside the
// chain (e.g. acquire on a write) cannot be strengthened meaningfully and
// are left alone.
static std::optional<omp::ClauseMemoryOrderKind>
nextStrongerOrder(Operation *innerOp, omp::ClauseMemoryOrderKind order,
                  std::mt19937 &rng) {
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
}

// Move one randomly chosen atomic read/write op one memory-order stage in
// the given direction. Returns false when no op qualifies.
static bool adjustMemoryOrder(func::FuncOp op, std::mt19937 &rng,
                              MemoryOrderDirection direction) {
    auto stepOrder = direction == MemoryOrderDirection::Weaker
                         ? nextWeakerOrder
                         : nextStrongerOrder;

    SmallVector<std::pair<Operation *, omp::ClauseMemoryOrderKind>> candidates;
    op.getBody().walk([&](Operation *innerOp) {
        auto order = getMemoryOrder(innerOp);
        if (!order || !isa<omp::AtomicWriteOp, omp::AtomicReadOp>(innerOp))
            return;
        if (auto next = stepOrder(innerOp, *order, rng))
            candidates.push_back({innerOp, *next});
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [target, newOrder] = candidates[dist(rng)];
    setMemoryOrder(target, newOrder);
    return true;
}

} // namespace

// randomly reorder eligible runs of loads in the function
// - runs inside threads only reorder loads of thread-local memrefs
// - runs in the single-threaded body may reorder loads of any memref
bool tryLoadReordering(func::FuncOp op, RewriterBase &rewriter,
                       std::mt19937 &rng) {
    ModuleOp module = op->getParentOfType<ModuleOp>();

    if (!module)
        return false;

    llvm::DenseSet<func::FuncOp> threadReachable;

    // We compute the set of functions reachable from thread regions in the module
    // This is used to determine if a function is executing inside a thread region or not
    // thus preventing reordering of loads that are not thread-local in a thread region
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
bool tryInsertFence(func::FuncOp op, RewriterBase &rewriter,
                    std::mt19937 &rng) {
    InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
    if (!point.op && !point.block)
        return false;
    setInsertPoint(rewriter, point);
    omp::FlushOp::create(rewriter, op.getLoc(), ValueRange{});
    return true;
}

// randomly remove fence
bool tryRemoveFence(func::FuncOp op, RewriterBase &rewriter,
                    std::mt19937 &rng) {
    SmallVector<omp::FlushOp> fences;
    op.walk([&](omp::FlushOp fence) { fences.push_back(fence); });
    if (fences.empty())
        return false;
    std::uniform_int_distribution<size_t> dist(0, fences.size() - 1);
    rewriter.eraseOp(fences[dist(rng)]);
    return true;
}

// Insert a dummy atomic write to a fresh thread-local and non-shared memory location
bool tryInsertAtomicWriteInThread(func::FuncOp op, RewriterBase &rewriter,
                                  std::mt19937 &rng) {
    InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
    if (!point.op && !point.block)
        return false;
    setInsertPoint(rewriter, point);

    Location loc = op.getLoc();

    // Dummy 0-d memref on this thread's stack and an atomic write of 0 to it
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

// Insert a dummy atomic read from a thread-local memref that is already in scope
bool tryInsertAtomicReadInThread(func::FuncOp op, RewriterBase &rewriter,
                                 std::mt19937 &rng) {
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

    // Finding a thread-local memref in scope at the insertion point
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

    // randomly pick one of the candidates and insert an atomic read from it to a fresh thread-local memref
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

// Insert a strong compare-and-swap on a fresh thread-local location
bool tryInsertAtomicCAS(func::FuncOp op, RewriterBase &rewriter,
                        std::mt19937 &rng) {
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
        /*memory_order=*/omp::ClauseMemoryOrderKindAttr{},
        /*fail_memory_order=*/omp::ClauseMemoryOrderKindAttr{});

    Block *body = rewriter.createBlock(&cas.getRegion(), {},
                                       {rewriter.getI32Type()}, {loc});

    rewriter.setInsertionPointToStart(body);

    Value expected = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(42));
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      body->getArgument(0), expected);
    Value desired = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(7));
    Value sel = arith::SelectOp::create(rewriter, loc, cmp, desired, body->getArgument(0));

    omp::YieldOp::create(rewriter, loc, ValueRange{sel});
    return true;
}

// Insert a no-op arithmetic operation on the stored value of a random atomic write
bool tryInsertReadArith(func::FuncOp op, RewriterBase &rewriter,
                        std::mt19937 &rng) {
    // get all atomic writes in the function and pick one at random
    SmallVector<omp::AtomicWriteOp> writes;
    op.getBody().walk([&](omp::AtomicWriteOp write) {
        writes.push_back(write);
    });

    if (writes.empty())
        return false;

    // only writes of an integer or index expression can carry the no-op arithmetic operation
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

// Duplicate a store to a thread-local memref that is only accessed in one omp.section
bool tryLocalStoreDuplication(func::FuncOp op, RewriterBase &rewriter,
                              std::mt19937 &rng) {
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

    // find all thread-local memref stores
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

// Randomly insert a fresh memref and a chain of arith ops inside a thread region
bool tryInsertRandomArith(func::FuncOp op, RewriterBase &rewriter,
                          std::mt19937 &rng) {
    InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
    if (!point.op && !point.block)
        return false;
    setInsertPoint(rewriter, point);

    ArithGenerator(rng).generate(rewriter, op.getLoc());

    return true;
}

// Randomly insert a fresh memref and a chain of loads/stores to it inside a thread region
bool tryInsertRandomMemref(func::FuncOp op, RewriterBase &rewriter,
                           std::mt19937 &rng) {
    InsertPoint point = makeThreadInsertPoint(op, rewriter, rng);
    if (!point.op && !point.block)
        return false;
    setInsertPoint(rewriter, point);

    ArithGenerator(rng).generateMemref(rewriter, op.getLoc());

    return true;
}

// Insert random-length thread-local delay chains into a gap between shared-memory ops
// Additionally used during agitation process for jittering
bool tryInsertJitter(func::FuncOp op, RewriterBase &rewriter,
                     std::mt19937 &rng) {
    SmallVector<InsertPoint> gaps;
    collectSharedMemoryGaps(op.getRegion(), /*inThread=*/false, gaps);

    InsertPoint point;
    if (!gaps.empty()) {
        std::uniform_int_distribution<size_t> dist(0, gaps.size() - 1);
        point = gaps[dist(rng)];
    } else {
        point = makeThreadInsertPoint(op, rewriter, rng);
    }
    if (!point.op && !point.block)
        return false;
    setInsertPoint(rewriter, point);

    ArithGenerator gen(rng);
    int chains = std::uniform_int_distribution<int>(1, 3)(rng);
    for (int i = 0; i < chains; ++i)
        gen.generateMixed(rewriter, op.getLoc());

    return true;
}

// Insert a no-op comparison of a thread-local integer value with itself
bool tryInsertComparison(func::FuncOp op, RewriterBase &rewriter,
                         std::mt19937 &rng) {
    // Collect integer-typed results defined inside thread regions
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

// Wrap thread-local ops in both arms of an scf.if, creating one when none exists
bool tryInsertBothArmsIf(func::FuncOp op, RewriterBase &rewriter,
                         std::mt19937 &rng) {
    Location loc = op.getLoc();

    // chains of consecutive wrappable ops inside an omp.section
    // results must be unused so they can be moved into the if without breaking SSA
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

// wrap a maximal run of consecutive wrappable ops inside an omp.section in an omp.critical
// ops inside critical regions are not eligible, so we don't place one critical inside another
bool tryInsertCriticalSection(func::FuncOp op, RewriterBase &rewriter,
                              std::mt19937 &rng) {
    SmallVector<SmallVector<Operation *>> chains =
        collectOpRuns(op, [](Operation *innerOp) {
            return innerOp->getParentOfType<omp::SectionOp>() &&
                   !innerOp->getParentOfType<omp::CriticalOp>() &&
                   isWrappable(innerOp);
        });

    // a chain that contains an omp.critical in any nested region would
    // place one critical inside another once wrapped
    llvm::erase_if(chains, [](const SmallVector<Operation *> &chain) {
        for (Operation *op : chain) {
            WalkResult result = op->walk([&](Operation *inner) {
                return isa<omp::CriticalOp>(inner)
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
            });
            if (result.wasInterrupted())
                return true;
        }
        return false;
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

// Insert a new omp.parallel after a randomly chosen existing one
// Should not affect any existing parallel ops
bool tryInsertParallelOp(func::FuncOp op, RewriterBase &rewriter,
                         std::mt19937 &rng) {
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

// Unroll single-threaded omp.parallel/omp.sections/omp.section ops
bool tryUnrollSingleThread(func::FuncOp op, RewriterBase &rewriter,
                           std::mt19937 &rng) {
    SmallVector<omp::ParallelOp> candidates;

    // Find all omp.parallel ops that contain a single omp.sections op
    // with a single omp.section op inside it, and that section has a single-block body.
    op.getBody().walk([&](omp::ParallelOp parallel) {
        Region &parallelRegion = parallel.getRegion();
        if (parallelRegion.getBlocks().size() != 1)
            return;
        Block &parallelBlock = parallelRegion.front();
        auto sectionsOps = llvm::to_vector(parallelBlock.getOps<omp::SectionsOp>());

        if (sectionsOps.size() != 1)
            return;

        omp::SectionsOp sections = sectionsOps[0];
        Region &sectionsRegion = sections.getRegion();

        if (sectionsRegion.getBlocks().size() != 1)
            return;

        Block &sectionsBlock = sectionsRegion.front();
        auto sectionOps = llvm::to_vector(sectionsBlock.getOps<omp::SectionOp>());
        if (sectionOps.size() != 1)
            return;

        candidates.push_back(parallel);
    });

    if (candidates.empty())
        return false;

    // Unroll every eligible parallel, move the single section's body to
    // where the omp.parallel op was, then remove the section, sections,
    // and parallel ops.
    bool unrolled = false;
    for (omp::ParallelOp parallel : candidates) {
        // re-locate the single omp.sections/omp.section inside the parallel
        omp::SectionsOp sections;
        parallel.walk([&](omp::SectionsOp s) { sections = s; });
        omp::SectionOp section;
        sections.walk([&](omp::SectionOp s) { section = s; });

        // Only unroll a bare nest
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

// Relax the memory order of a randomly chosen atomic omp op
bool tryRelaxOperation(func::FuncOp op, RewriterBase &,
                       std::mt19937 &rng) {
    return adjustMemoryOrder(op, rng, MemoryOrderDirection::Weaker);
}

// Restrict the memory order of a randomly chosen atomic omp op
bool tryRestrictOperation(func::FuncOp op, RewriterBase &,
                          std::mt19937 &rng) {
    return adjustMemoryOrder(op, rng, MemoryOrderDirection::Stronger);
}

// Insert a fence inbetween two consecutive atomic reads/writes in a thread region
bool tryInsertFenceInbetweenMemOps(func::FuncOp op, RewriterBase &rewriter,
                                   std::mt19937 &rng) {
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

// Insert a fence before and/or after a randomly chosen atomic read/write/compare op with seq_cst memory order
bool tryInsertFenceAroundSeqCst(func::FuncOp op, RewriterBase &rewriter,
                                std::mt19937 &rng) {
    SmallVector<Operation *> candidates;
    op.getBody().walk([&](Operation *innerOp) {
        std::optional<omp::ClauseMemoryOrderKind> order;
        if (auto write = dyn_cast<omp::AtomicWriteOp>(innerOp))
            order = write.getMemoryOrder();
        else if (auto read = dyn_cast<omp::AtomicReadOp>(innerOp))
            order = read.getMemoryOrder();
        else if (auto cmp = dyn_cast<omp::AtomicCompareOp>(innerOp))
            order = cmp.getMemoryOrder();
        else
            return;

        if (order.value_or(omp::ClauseMemoryOrderKind::Seq_cst) ==
            omp::ClauseMemoryOrderKind::Seq_cst)
            candidates.push_back(innerOp);
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    Operation *target = candidates[dist(rng)];

    bool before = std::bernoulli_distribution(0.5)(rng);
    bool after = std::bernoulli_distribution(0.5)(rng);
    if (!before && !after)
        before = true;

    Location loc = op.getLoc();
    if (before) {
        rewriter.setInsertionPoint(target);
        omp::FlushOp::create(rewriter, loc, ValueRange{});
    }
    if (after) {
        rewriter.setInsertionPointAfter(target);
        omp::FlushOp::create(rewriter, loc, ValueRange{});
    }
    return true;
}

namespace {
const FunctionTransform kGenericTransforms[] = {
    FunctionTransform("insert-fence", "generic", &tryInsertFence,
                      mlir_mracle::OutcomeRelation::Subset),
    FunctionTransform("remove-fence", "generic", &tryRemoveFence,
                      mlir_mracle::OutcomeRelation::Superset),
    FunctionTransform("insert-atomic-cas", "generic", &tryInsertAtomicCAS,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-atomic-write", "generic",
                      &tryInsertAtomicWriteInThread,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-atomic-read", "generic",
                      &tryInsertAtomicReadInThread,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-read-arith", "generic", &tryInsertReadArith,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-random-arith", "generic", &tryInsertRandomArith,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-random-memref", "generic", &tryInsertRandomMemref,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-jitter", "generic", &tryInsertJitter,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("local-store-duplication", "generic",
                      &tryLocalStoreDuplication,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-comparison", "generic", &tryInsertComparison,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-both-arms-if", "generic", &tryInsertBothArmsIf,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-parallel", "generic", &tryInsertParallelOp,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("load-reordering", "generic", &tryLoadReordering,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("relax-operation", "generic", &tryRelaxOperation,
                      mlir_mracle::OutcomeRelation::Superset),
    FunctionTransform("restrict-operation", "generic", &tryRestrictOperation,
                      mlir_mracle::OutcomeRelation::Subset),
    FunctionTransform("insert-fence-between-mem-ops", "generic",
                      &tryInsertFenceInbetweenMemOps,
                      mlir_mracle::OutcomeRelation::Subset),
    FunctionTransform("insert-fence-around-seq-cst", "generic",
                      &tryInsertFenceAroundSeqCst,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("unroll-single-thread", "generic", &tryUnrollSingleThread,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("insert-critical", "generic", &tryInsertCriticalSection,
                      mlir_mracle::OutcomeRelation::Subset),
};
} // namespace

llvm::ArrayRef<FunctionTransform> getGenericTransforms() {
    return kGenericTransforms;
}

} // namespace metamorphic
} // namespace mlir
