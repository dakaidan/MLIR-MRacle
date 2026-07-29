#pragma once

#include "conquer/core/types.h"
#include "conquer/core/tensor.h"

#include <mlir/Dialect/Tosa/Transforms/Passes.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Format.h>

#include <iomanip>
#include <string>
#include <vector>

namespace conquer {
class ModelSession {
  public:
    std::vector<TensorConstraint> expectedInputs;
    std::vector<TensorConstraint> expectedOutputs;

    /// Stores a reference to the module.
    /// NOTE: The module must stay alive as long as this Session exists.
    void load(mlir::ModuleOp mod);

    /// Step 1: Validate that the provided inputs match the model's requirements.
    /// This checks Data Types and Ranks. For fixed dimensions, it checks equality.
    /// For dynamic dimensions (-1), it allows any size.
    [[nodiscard]] bool validateInputs(const std::vector<TensorView> &inputs) const;

    /// Step 2: Calculate the exact output shapes for a specific run.
    /// If the model is static, this is instant.
    /// If the model is dynamic, this runs a TOSA shape inference pass.
    /// @return A vector of shapes (std::vector<int64_t>) corresponding to each output.
    std::vector<std::vector<int64_t>> resolveOutputShapes(const std::vector<TensorView> &inputs);

    [[nodiscard]] OutputAllocation allocateOutputs(const std::vector<std::vector<int64_t>> &shapes) const;

  private:
    mlir::ModuleOp module;

    /// The expensive path: Clone module -> Specialise -> Infer -> Read Returns
    std::vector<std::vector<int64_t>> runShapeInference(const std::vector<TensorView> &inputs);

    static void parseTypeList(mlir::TypeRange types, std::vector<TensorConstraint> &constraints,
                              const std::string &prefix);

    static bool validateList(const std::vector<TensorView> &views, const std::vector<TensorConstraint> &constraints,
                             const std::string &context);

    static DataType mapMlirType(const mlir::Type type);
};
} // namespace conquer