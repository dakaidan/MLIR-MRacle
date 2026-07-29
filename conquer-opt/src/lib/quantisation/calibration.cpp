#include "conquer/quantisation/calibration.h"
#include "conquer/core/logging.h"

#include "conquer/backend/cpu/cpu_target.h"
#include "conquer/passes/analysis/activation_annotation_pass.h"
#include "conquer/passes/analysis/activation_instrumentation_pass.h"
#include "conquer/passes/analysis/weight_analysis_pass.h"
#include "conquer/passes/runner.h"

#include <llvm/Support/Debug.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-calibration"

static std::vector<conquer::CalibrationCollector>* g_collectors = nullptr;
static std::vector<conquer::SensitivityCollector>* g_sens_collectors = nullptr;

extern "C" {
    void _mlir_ciface_conquer_record_activation_stats(int32_t layer_id, void* raw_unranked_memref) {
        if (!g_collectors || layer_id < 0 || static_cast<std::size_t>(layer_id) >= g_collectors->size()) return;

        auto* unranked_memref = static_cast<UnrankedMemRefType<float>*>(raw_unranked_memref);
        const DynamicMemRefType<float> memref(*unranked_memref);

        int64_t num_elements = 1;
        for (int i = 0; i < memref.rank; ++i) {
            num_elements *= memref.sizes[i];
        }

        if (num_elements == 0) return;

        // Min/Max/Histogram Collection
        const std::vector<float> values(memref.data, memref.data + num_elements);
        (*g_collectors)[layer_id].updateStreaming(values);

        // Sensitivity (Entropy) Collection
        if (!g_sens_collectors || layer_id < 0 || static_cast<std::size_t>(layer_id) >= g_sens_collectors->size()) return;

        // Skip scalars (rank 0), as covariance/entropy of a scalar makes no sense.
        if (memref.rank == 0) return;

        // Determine the Channel/Feature dimension (C)
        // Heuristic: TOSA operations natively output features on the innermost (last) dimension.
        int64_t c_dim = memref.sizes[memref.rank - 1];

        // Sanity check: If the channel dimension is 1, entropy will be 0 anyway,
        // but we still process it to avoid breaking the collector's initialization state.
        if (c_dim > 0) {
            // Because C is the innermost dimension, the data is laid out perfectly
            // as contiguous C-dimensional vectors. We can safely pass 'values' directly.
            (*g_sens_collectors)[layer_id].updateStreaming(values, c_dim);
        }
    }
}

void conquer::calibrate_module(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs) {
    L_INFO("Starting module calibration.");
    _weight_calibration(module);
    _activation_calibration(module, calibration_inputs);
    L_INFO("Module calibration complete.");
}

void conquer::_weight_calibration(mlir::Operation *module) {
    L_INFO("Starting weight calibration.");
    conquer::runPassOnModule<WeightAnalysisPass>(module);
    L_INFO("Weight calibration complete.");
}

void conquer::_activation_calibration(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs) {
    mlir::Operation *moduleClone = module->clone();
    conquer::runPassOnModule<ActivationInstrumentationPass>(moduleClone);
    const auto [calibration, sensitivity] = run_calibration(moduleClone, calibration_inputs);
    conquer::runPassOnModule<ActivationAnnotationPass>(module, calibration, sensitivity);
    moduleClone->erase();
}

std::pair<conquer::CalibrationResult, conquer::SensitivityResult> conquer::run_calibration(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs) {
    auto session = ModelSession();
    session.load(mlir::cast<mlir::ModuleOp>(module));

    size_t num_instrumented_layers = 0;
    module->walk([&](mlir::func::CallOp callOp) {
        if (callOp.getCallee() == "conquer_record_activation_stats") {
            num_instrumented_layers++;
        }
    });

    std::vector<CalibrationCollector> collectors(num_instrumented_layers);
    g_collectors = &collectors;

    std::vector<SensitivityCollector> sens_collectors(num_instrumented_layers);
    g_sens_collectors = &sens_collectors;

    const auto target = std::make_unique<CPUTarget>();

    if (auto err = target->compile(module->clone())) {
        g_collectors = nullptr;
        g_sens_collectors = nullptr;
        throw std::runtime_error("Failed to lower model for CPU target: " + toString(std::move(err)));
    }

    for (const auto &alloc : calibration_inputs) {
        auto input_view = alloc.getView();

        if (!session.validateInputs({input_view})) {
            g_collectors = nullptr;
            g_sens_collectors = nullptr;
            throw std::runtime_error("Calibration input does not match expected constraints.");
        }

        auto [buffers, views] = session.allocateOutputs(session.resolveOutputShapes({input_view}));

        if (auto err = target->execute({input_view}, views)) {
            g_collectors = nullptr;
            g_sens_collectors = nullptr;
            throw std::runtime_error("Failed to execute: " + toString(std::move(err)));
        }
    }

    CalibrationResult results;
    results.reserve(collectors.size());
    for (auto &collector : collectors) {
        results.push_back(collector.getStats());
    }

    SensitivityResult sensitivity_results;
    sensitivity_results.reserve(sens_collectors.size());
    for (auto &collector : sens_collectors) {
        sensitivity_results.push_back(collector.getStats());
    }

    g_collectors = nullptr;
    g_sens_collectors = nullptr;

    return {results, sensitivity_results};
}