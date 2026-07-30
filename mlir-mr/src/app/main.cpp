#include <mlir/Dialect/Arith/IR/Arith.h>
#include <iostream>

int main() {
    mlir::MLIRContext ctx;

    ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
    std::cout << "Successfully compiled with MLIR." << std::endl;
    return 0;
}