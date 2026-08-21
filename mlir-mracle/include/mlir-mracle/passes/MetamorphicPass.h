#pragma once
#include <memory>
#include <string>

namespace mlir_mracle {
struct RunInfo;
}

namespace mlir {

class Pass;
std::unique_ptr<Pass> createMetamorphicPass(
    int seed = 42, mlir_mracle::RunInfo *runInfo = nullptr,
    std::string transform = "", int maxApply = 1);

}
