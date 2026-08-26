#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include <random>

namespace mlir {
namespace metamorphic {

// Built-in generic transforms (target "generic"): valid on every memory
// model. Declared here so external code can wrap them in a FunctionTransform
// subclass and override the outcome relation.
bool tryInsertFence(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryRemoveFence(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertAtomicCAS(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertAtomicWriteInThread(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertAtomicReadInThread(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertReadArith(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertRandomArith(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertRandomMemref(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertJitter(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryLocalStoreDuplication(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertComparison(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertBothArmsIf(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertParallelOp(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryLoadReordering(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryRelaxOperation(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryRestrictOperation(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertFenceInbetweenMemOps(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertFlushAroundSeqCst(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryUnrollSingleThread(func::FuncOp, RewriterBase &, std::mt19937 &);
bool tryInsertCriticalSection(func::FuncOp, RewriterBase &, std::mt19937 &);

} // namespace metamorphic
} // namespace mlir
