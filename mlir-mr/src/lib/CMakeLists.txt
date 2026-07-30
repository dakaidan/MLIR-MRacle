# Header/source-free INTERFACE library while src/lib/ is empty.
# Switch to STATIC and add *.cpp once the first implementation lands.
add_library(mlir_mr INTERFACE)
add_library(MLIRMR::mlir_mr ALIAS mlir_mr)

set_property(TARGET mlir_mr PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mlir_mr INTERFACE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRTransforms
    MLIRTosaDialect
    MLIRArithDialect
)
