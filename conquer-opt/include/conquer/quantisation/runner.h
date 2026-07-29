#pragma once
#include "conquer/core/utils.h"
#include "conquer/passes/analysis/strip_annotations_pass.h"
#include "conquer/passes/naming_pass.h"
#include "conquer/passes/quantisation/float_quantisation_pass.h"
#include "conquer/passes/quantisation/integer_quantisation_pass.h"
#include "conquer/passes/quantisation/shared/quantisation_canonicalisation_pass.h"
#include "conquer/passes/runner.h"
#include "conquer/quantisation/calibration.h"
#include "conquer/quantisation/policy.h"
#include "conquer/passes/analysis/externalise_constants_pass.h"

#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_ostream.h>

#include <string>

#include "conquer/passes/quantisation/lower_quantisation_pass.h"

namespace conquer {
inline void quantise(mlir::Operation *module, const QuantisationPolicy &policy,
                     const std::vector<TensorAllocation> &calibrationData = {}) {
    runPassOnModule<ModelNamingPass>(module);

    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);
    module->print(rso);

    const auto [is_cached, result] = cache_result(
        moduleStr,
        [&]() -> std::string {
            calibrate_module(module, calibrationData);
            std::string calibrated_string;
            llvm::raw_string_ostream calibrated_rso(calibrated_string);
            module->print(calibrated_rso);
            return calibrated_string;
        },
        "calibration");

    if (is_cached) {
        mlir::MLIRContext *context = module->getContext();
        mlir::OwningOpRef<mlir::ModuleOp> cachedModule = mlir::parseSourceString<mlir::ModuleOp>(result, context);
        if (!cachedModule) {
            throw std::runtime_error("Failed to parse cached calibrated module.");
        }

        auto modOp = mlir::cast<mlir::ModuleOp>(module);
        modOp.getBodyRegion().takeBody(cachedModule->getBodyRegion());
    }

    mlir::PassManager pm(module->getContext());
    pm.enableVerifier(true);
    pm.addPass(createFloatQuantisationPass(policy));
    pm.addPass(createQuantisationCanonicalisationPass(policy));
    pm.addPass(createIntegerQuantisationPass(policy));
    pm.addPass(createQuantisationCanonicalisationPass(policy));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createLowerQuantisationPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createExternaliseConstantsPass());
    pm.addPass(createStripAnnotationPass());
    if (mlir::failed(pm.run(mlir::cast<mlir::ModuleOp>(module)))) {
        module->emitError("Quantisation failed during pass execution.");
        throw std::runtime_error("Quantisation failed during pass execution.");
    }
}
} // namespace conquer