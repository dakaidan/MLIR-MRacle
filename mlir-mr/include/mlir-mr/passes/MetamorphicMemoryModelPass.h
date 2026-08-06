#pragma once
#include <memory>
#include <string>

namespace mlir {

class Pass;
std::unique_ptr<Pass> createMetamorphicMemoryModelPass(
    int seed = 42, const std::string &transforms = "", bool debug = false);

}