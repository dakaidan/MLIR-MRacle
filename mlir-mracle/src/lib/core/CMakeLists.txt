add_library(mlir_mracle_core STATIC types.cpp)
add_library(MLIRMRACLE::mlir_mracle_core ALIAS mlir_mracle_core)

set_property(TARGET mlir_mracle_core PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_core PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_core PUBLIC
    llvm_mlir_headers
    LLVMSupport
)
