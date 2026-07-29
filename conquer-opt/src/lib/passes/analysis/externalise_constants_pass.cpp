#include "conquer/passes/analysis/externalise_constants_pass.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <cstdlib>
#include <cstring>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-externalise-constants-pass"

void conquer::ExternaliseConstantsPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();

    uint64_t resourceCounter = 0;

    module.walk([&](mlir::Operation *op) {
        mlir::Attribute baseAttr;

        // Target both TOSA and Arith constants, capturing them as generic Attributes
        if (auto tosaConst = llvm::dyn_cast<mlir::tosa::ConstOp>(op)) {
            baseAttr = static_cast<mlir::Attribute>(tosaConst.getValues());
        } else if (auto arithConst = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
            baseAttr = static_cast<mlir::Attribute>(arithConst.getValue());
        }

        if (!baseAttr) return;

        // If it's already a resource, or an opaque/splat attr we can't extract cleanly, skip it.
        const auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(baseAttr);
        if (!denseAttr || denseAttr.isSplat()) return;

        // Skip small tensors to avoid bloating the resource dictionary with scalars/tiny arrays.
        if (denseAttr.getNumElements() < 16) return;

        const auto tensorType = llvm::cast<mlir::RankedTensorType>(denseAttr.getType());
        const llvm::ArrayRef<char> rawData = denseAttr.getRawData();

        // Manually allocate and copy the data to bypass MLIR's volatile heap managers
        void* copiedData = std::malloc(rawData.size());
        std::memcpy(copiedData, rawData.data(), rawData.size());

        auto blob = mlir::AsmResourceBlob(
            llvm::ArrayRef<char>(static_cast<char*>(copiedData), rawData.size()),
            /*alignment=*/8,
            /*deleter=*/[copiedData](void*, size_t, size_t) { std::free(copiedData); },
            /*dataIsMutable=*/false
        );

        const std::string resourceName = "blob_" + std::to_string(resourceCounter++);

        // Use the MLIR 21 3-argument overload: (ShapedType, StringRef, AsmResourceBlob)
        const auto resourceAttr = mlir::DenseResourceElementsAttr::get(
            tensorType,
            resourceName,
            std::move(blob)
        );

        // Safely iterate by const reference to avoid object slicing warnings
        for (const mlir::NamedAttribute &attr : op->getAttrs()) {
            if (attr.getValue() == baseAttr) {
                op->setAttr(attr.getName(), resourceAttr);
                break;
            }
        }
    });
}