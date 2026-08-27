#include "mlir-mracle/passes/transforms/arm/ArmTransforms.h"
#include "mlir-mracle/passes/transforms/Transforms.h"
#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "mlir-mracle/context/context.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <random>
#include <utility>

namespace mlir {
namespace metamorphic {

// reorders two relaxed read and write operations in a function, if possible, to commute them
bool tryCommuteRelaxedReadWrite(func::FuncOp op, RewriterBase &rewriter,
                                std::mt19937 &rng) {
    SmallVector<std::pair<omp::AtomicReadOp, omp::AtomicWriteOp>> candidates;
    op.walk([&](omp::AtomicReadOp read) {
        Operation *nextOp = read->getNextNode();
        while (nextOp) {
            if (auto nextWrite = dyn_cast<omp::AtomicWriteOp>(nextOp)) {
                candidates.push_back({read, nextWrite});
                break;
            }
            // allow a non-atomic op in between only when it is a pure arith
            // op, since it does not touch memory and cannot observe the
            // values being reordered
            if (isa<memref::LoadOp, memref::StoreOp,
                    omp::AtomicReadOp>(nextOp))
                break;
            if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                     arith::ConstantOp, arith::CmpIOp>(nextOp))
                break;
            nextOp = nextOp->getNextNode();
        }
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [read, write] = candidates[dist(rng)];

    // reorder to make the write execute before the read
    rewriter.moveOpBefore(write, read);

    return true;
}

// reorders two relaxed write operations in a function, if possible, to commute them
bool tryCommuteRelaxedWriteWrite(func::FuncOp op, RewriterBase &rewriter,
                                 std::mt19937 &rng) {
    SmallVector<std::pair<omp::AtomicWriteOp, omp::AtomicWriteOp>> candidates;
    op.walk([&](omp::AtomicWriteOp first) {
        Operation *nextOp = first->getNextNode();
        while (nextOp) {
            if (auto nextWrite = dyn_cast<omp::AtomicWriteOp>(nextOp)) {
                candidates.push_back({first, nextWrite});
                break;
            }
            // allow a non-atomic op in between only when it is a pure arith
            // op, since it does not touch memory and cannot observe the
            // values being reordered
            if (isa<memref::LoadOp, memref::StoreOp,
                    omp::AtomicReadOp>(nextOp))
                break;
            if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                     arith::ConstantOp, arith::CmpIOp>(nextOp))
                break;
            nextOp = nextOp->getNextNode();
        }
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [first, second] = candidates[dist(rng)];

    // reorder to make the second write execute before the first
    rewriter.moveOpBefore(second, first);

    return true;
}

namespace {
const FunctionTransform kArmTransforms[] = {
    FunctionTransform("commute-relaxed-read-write", "armv8",
                      &tryCommuteRelaxedReadWrite,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("commute-relaxed-write-write", "armv8",
                      &tryCommuteRelaxedWriteWrite,
                      mlir_mracle::OutcomeRelation::Equality),
};
} // namespace

llvm::ArrayRef<FunctionTransform> getArmTransforms() {
    return kArmTransforms;
}

} // namespace metamorphic
} // namespace mlir
