#pragma once

#include "conquer/runtime/session.h"

#include <vector>

#include "stats.h"

namespace conquer {
using CalibrationResult = std::vector<CalibrationStats>;
using SensitivityResult = std::vector<SensitivityStats>;

void calibrate_module(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs);

void _weight_calibration(mlir::Operation *module);
void _activation_calibration(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs);

std::pair<CalibrationResult, SensitivityResult> run_calibration(mlir::Operation *module, const std::vector<TensorAllocation> &calibration_inputs);
} // namespace conquer