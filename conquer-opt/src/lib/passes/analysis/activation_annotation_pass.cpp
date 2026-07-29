#include "conquer/passes/analysis/activation_annotation_pass.h"
#include "conquer/quantisation/stats.h"
#include "conquer/core/logging.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-activation-annotation"

void conquer::ActivationAnnotationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(&getContext());

    size_t statIndex = 0;

    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() == "conquer_record_activation_stats") return;

        std::vector<mlir::Value> valuesToAnnotate;

        if (!funcOp.getBody().empty()) {
            for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
                if (llvm::isa<mlir::TensorType>(funcOp.getArgument(i).getType())) {
                    valuesToAnnotate.push_back(funcOp.getArgument(i));
                }
            }
        }

        funcOp.walk([&](mlir::Operation *op) {
            if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return;

            // MUST MIRROR THE FOOLPROOF FIX HERE
            if (op->hasTrait<mlir::OpTrait::ConstantLike>()) return;

            if (op->getNumResults() == 0) return;

            if (llvm::isa<mlir::TensorType>(op->getResult(0).getType())) {
                valuesToAnnotate.push_back(op->getResult(0));
            }
        });

        for (mlir::Value val : valuesToAnnotate) {
            if (statIndex >= activationStats.size() || statIndex >= sensitivityStats.size()) {
                funcOp->emitWarning("Ran out of stats during annotation.");
                return;
            }

            const auto &[min, max, emaMin, emaMax, pctMin, pctMax, klInt8Min, klInt8Max, klInt4Min, klInt4Max] =
                activationStats[statIndex];
            const float entropySens = sensitivityStats[statIndex].entropySensitivity;
            statIndex++;

            if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
                unsigned argNum = arg.getArgNumber();
                funcOp.setArgAttr(argNum, "activation_stats.min_max.min", builder.getF32FloatAttr(min));
                funcOp.setArgAttr(argNum, "activation_stats.min_max.max", builder.getF32FloatAttr(max));
                funcOp.setArgAttr(argNum, "activation_stats.ema.min", builder.getF32FloatAttr(emaMin));
                funcOp.setArgAttr(argNum, "activation_stats.ema.max", builder.getF32FloatAttr(emaMax));
                funcOp.setArgAttr(argNum, "activation_stats.percentile.min", builder.getF32FloatAttr(pctMin));
                funcOp.setArgAttr(argNum, "activation_stats.percentile.max", builder.getF32FloatAttr(pctMax));
                funcOp.setArgAttr(argNum, "activation_stats.kl.int8.min", builder.getF32FloatAttr(klInt8Min));
                funcOp.setArgAttr(argNum, "activation_stats.kl.int8.max", builder.getF32FloatAttr(klInt8Max));
                funcOp.setArgAttr(argNum, "activation_stats.kl.int4.min", builder.getF32FloatAttr(klInt4Min));
                funcOp.setArgAttr(argNum, "activation_stats.kl.int4.max", builder.getF32FloatAttr(klInt4Max));

                funcOp.setArgAttr(argNum, "sensitivity_value.entropy.activation", builder.getF32FloatAttr(entropySens));
            } else {
                mlir::Operation *op = val.getDefiningOp();
                op->setAttr("activation_stats.min_max.min", builder.getF32FloatAttr(min));
                op->setAttr("activation_stats.min_max.max", builder.getF32FloatAttr(max));
                op->setAttr("activation_stats.ema.min", builder.getF32FloatAttr(emaMin));
                op->setAttr("activation_stats.ema.max", builder.getF32FloatAttr(emaMax));
                op->setAttr("activation_stats.percentile.min", builder.getF32FloatAttr(pctMin));
                op->setAttr("activation_stats.percentile.max", builder.getF32FloatAttr(pctMax));
                op->setAttr("activation_stats.kl.int8.min", builder.getF32FloatAttr(klInt8Min));
                op->setAttr("activation_stats.kl.int8.max", builder.getF32FloatAttr(klInt8Max));
                op->setAttr("activation_stats.kl.int4.min", builder.getF32FloatAttr(klInt4Min));
                op->setAttr("activation_stats.kl.int4.max", builder.getF32FloatAttr(klInt4Max));

                op->setAttr("sensitivity_value.entropy.activation", builder.getF32FloatAttr(entropySens));
            }
        }
    });

    if (statIndex < activationStats.size() || statIndex < sensitivityStats.size()) {
        L_DEBUG("Warning: some stats were left unused.");
    }
}