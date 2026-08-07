#pragma once
#include <memory>
#include <string>

namespace mlir_mr {
struct RunInfo;
}

namespace mlir {

class Pass;
std::unique_ptr<Pass> createMetamorphicMemoryModelPass(
    int seed = 42, mlir_mr::RunInfo *runInfo = nullptr,
    std::string transform = "");

}
